import argparse
import math
from pathlib import Path
import random
import os
import sys
import cv2

import numpy as np
import neural_renderer as nr
import torch
from torch import Tensor, nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import yaml
import tsgan
from tsgan.render import NeuralRenderer

try:
    import wandb

except ImportError:
    wandb = None

from tsgan.utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from tsgan.models.op import conv2d_gradfix
from tsgan.models.stylegan2 import Generator, Discriminator
from tsgan.models.classifer import resnet50

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]                         # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))                 # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative

from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import non_max_suppression, scale_boxes

cstr = lambda s: f"\033[01;32m{s}\033[0m"


def train(args):
    args.npoint = 12306
    args.num_feature = 4
    args.latent_dim = 512

    (
        neural_renderer,
        encoder,
        decoder,
        cond_model,
        detector,
        detector_eval,
        compute_detector_loss,
        generator,
        discriminator,
        g_ema,
        g_optim,
        d_optim,
        loader,
        mix_train_set,
        device,
    ) = prepare_training(args)

    with torch.no_grad():
        real_texture = neural_renderer.textures
        texture_latent = encoder.forward(real_texture)
        batch_real_texture = real_texture.repeat(args.batch, *[1] * (len(real_texture.size()) - 1))
        batch_texture_latent = texture_latent.repeat(args.batch, *[1] * (len(texture_latent.size()) - 1))

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    # ----------------------------------------------
    # ---> start training
    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5**(32 / (10 * 1000))
    r_t_stat = 0

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    # sample_save_dir = os.path.join("tmp", args.save_dir, "sample")
    os.makedirs(sample_save_dir := os.path.join("tmp", args.save_dir, "sample"), exist_ok=True)
    # checkpoint_save_dir = os.path.join("tmp", args.save_dir, "checkpoint")
    os.makedirs(checkpoint_save_dir := os.path.join("tmp", args.save_dir, "checkpoint"), exist_ok=True)

    loader = sample_data(loader)
    for idx in pbar:
        i = idx + args.start_iter
        batch_data = next(loader)

        cond_images = batch_data["coco"]["image"].to(device)
        coco_label = batch_data["coco"]["predict_id"].to(device)

        carla_scene_images = batch_data["carla"]["image"]
        carla_render_params = batch_data["carla"]["render_param"]

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # ----------------------------------------
        # perform backward
        # ----------------------------------------
        texture_latent_mix_noise = batch_texture_latent.clone()
        for i_b in range(args.batch):
            noise_rotio = 0.5
            noise = torch.randn_like(texture_latent_mix_noise[i_b])
            texture_latent_mix_noise[i_b] = (1 - noise_rotio) * texture_latent_mix_noise[i_b] + noise_rotio * noise

        with torch.no_grad():
            cond_latent = cond_model.forward_latent(cond_images)

        # ----------------------------------------
        # perform backward
        # ----------------------------------------
        fake_texture_latent = generator.forward(texture_latent_mix_noise, cond_latent) # TODO: Inference

        real_pred = discriminator.forward(batch_texture_latent, cond_latent)
        fake_pred = discriminator.forward(fake_texture_latent, cond_latent)

        if i % args.d_loss_every == 0:
            # D logistic loss
            # d_loss = d_logistic_loss(real_pred, fake_pred)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            loss_dict["d"] = d_loss
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()

            # D backward
            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()

        # ----------------------------------------
        #   D regularization for every d_reg_every iterations
        # ----------------------------------------
        if i % args.d_reg_every == 0:
            batch_texture_latent.requires_grad = True
            real_pred = discriminator(batch_texture_latent, cond_latent)
            r1_loss = d_r1_loss(real_pred, batch_texture_latent)
            total_reg_loss = args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]
            discriminator.zero_grad()
            total_reg_loss.backward()
            d_optim.step()
            batch_texture_latent.requires_grad = False

        loss_dict["r1"] = r1_loss

        # frozen D
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        # G backward
        # fake_texture = generator.forward(texture_latent_mix_noise, cond_latent)
        fake_texture_latent = generator.forward(texture_latent_mix_noise, cond_latent) # TODO: Inference
        fake_pred = discriminator(fake_texture_latent, cond_latent)

        det_loss, lbox, lobj, lcls = 0, 0, 0, 0
        if i % args.g_det_every == 0:
            with torch.no_grad():
                fake_textures = decoder.forward(fake_texture_latent)

            render_image_list = []
            render_scene_list = []
            render_label_list = []
            scene_label_list = [] # scene image with label to check
            for i_b in range(fake_textures.size(0)):
                carla_scene_image = carla_scene_images[i_b]
                carla_render_param = carla_render_params[i_b]
                fake_texture = fake_textures[i_b].unsqueeze(0)
                tm = neural_renderer.textures_mask
                render_texture = tm * fake_texture + (1 - tm) * neural_renderer.textures

                neural_renderer.set_render_perspective(
                    carla_render_param["camera_transform"], carla_render_param["vehicle_transform"],
                    carla_render_param["fov"]
                )
                (rgb_images, depth_images, alpha_images) = neural_renderer.renderer.forward(
                    neural_renderer.vertices, neural_renderer.faces, torch.tanh(render_texture)
                )

                t_rgb_image = rgb_images.squeeze(0)
                t_alpha_image = alpha_images
                t_scene_image = torch.from_numpy(carla_scene_image).to(t_rgb_image.device).permute(2, 0, 1).float() / 255.  # yapf: disable
                t_render_image = t_alpha_image * t_rgb_image + (1 - t_alpha_image) * t_scene_image

                render_npimg: np.ndarray = t_rgb_image.detach().cpu().numpy() * 255
                render_npimg = render_npimg.astype(np.uint8).transpose(1, 2, 0)

                scene_npimg: np.ndarray = t_render_image.detach().cpu().numpy() * 255
                scene_npimg = scene_npimg.astype(np.uint8).transpose(1, 2, 0)
                scene_npimg = np.ascontiguousarray(scene_npimg)

                # if i_b == 0:
                #     cv2.imwrite(os.path.join(sample_save_dir, f'render.png'), render_npimg)

                render_image_list.append(t_rgb_image.unsqueeze(0))
                render_scene_list.append(t_render_image.unsqueeze(0))

                # find object label
                ret, binary = cv2.threshold(cv2.cvtColor(render_npimg, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                find_boxes = []
                for c in contours:
                    [x, y, w, h] = cv2.boundingRect(c)
                    find_boxes.append([x, y, x + w, y + h])
                fc = np.array(find_boxes)
                box = [min(fc[:, 0]), min(fc[:, 1]), max(fc[:, 2]), max(fc[:, 3])] # [x1,y1,x2,y2]

                if False:
                    [x1, y1, x2, y2] = box
                    cv2.rectangle(scene_npimg, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = mix_train_set.COCO_CLASS[int(coco_label[i_b])]
                    cv2.putText(scene_npimg, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imwrite(os.path.join(sample_save_dir, f'render.png'), scene_npimg)
                    # cv2.imwrite(os.path.join(sample_save_dir, f'render-{idx}.png'), scene_npimg)

                # YOLOv5 label: [image_idx, class, x_center, y_center, width, height]
                # why image_idx: [#253](https://github.com/ultralytics/yolov3/issues/253)
                # see [collate_fn](https://github.com/ultralytics/yolov5/blob/8ecc7276ecdd9c409b3dc8b9051142569009c6f4/utils/dataloaders.py#LL890C17-L890C21)
                c, h, w = t_rgb_image.shape
                render_label_list.append(torch.tensor([
                    i_b, int(coco_label[i_b]),
                    (box[0] + box[2]) / 2 / w,
                    (box[1] + box[3]) / 2 / h,
                    (box[2] - box[0]) / w,
                    (box[3] - box[1]) / h,
                ])) # yapf:disable
            render_images = torch.cat(render_image_list, dim=0).to(device)
            render_scenes = torch.cat(render_scene_list, dim=0).to(device)
            render_labels = torch.stack(render_label_list, dim=0).to(device)

            # with torch.no_grad():
            # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
            # https://www.zhihu.com/question/422373907/answer/1545222557
            # https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py#L67
            pred = detector.forward(render_images)
            # pred = [p.detach() + render_images - render_images.detach() for p in pred]
            # pred.grad_fn = fake_textures.grad_fn
            # for p in pred:
            #     p.requires_grad = True
            (det_loss, (lbox, lobj, lcls)) = compute_detector_loss.__call__(pred, render_labels)
            det_loss = 0.05 * lbox + 1.0 * lobj + 0.5 * lcls
            # det_loss.backward(retain_graph=True)
        loss_dict["lbox"] = lbox
        loss_dict["lobj"] = lobj
        loss_dict["lcls"] = lcls
        loss_dict["det"] = det_loss

        g_gen_loss = g_nonsaturating_loss(fake_pred)
        g_loss = g_gen_loss + 50 * det_loss
        loss_dict["g"] = g_gen_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # ----------------------------------------
        #   G regularization for every g_reg_every iterations
        # ----------------------------------------
        # if i % args.g_reg_every == 0:
        #     fake_texture_latent, latents = generator.forward(texture_latent_mix_noise, cond_latent, return_latents=True)
        #     path_loss, mean_path_length, path_lengths = g_path_regularize(
        #         fake_texture_latent, latents, mean_path_length
        #     )
        #     generator.zero_grad()

        #     weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss
        #     # if args.path_batch_shrink:
        #     #     weighted_path_loss += 0 * fake_texture[0, 0, 0, 0]
        #     weighted_path_loss.backward()
        #     g_optim.step()
        #     mean_path_length_avg = (reduce_sum(mean_path_length).item() / get_world_size())
        # loss_dict["path"] = path_loss
        # loss_dict["path_length"] = path_lengths.mean()

        # ----------------------------------------
        #  Valid
        # ----------------------------------------
        if (i % args.valid_every) == 0 and (render_scenes is not None):
            with torch.no_grad():
                pred = detector_eval.forward(render_scenes)[0]
            conf_thres, iou_thres = 0.25, 0.6
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False)
            result_imgs = []
            for i_b, det in enumerate(pred):
                cv2_img = np.ascontiguousarray(render_scenes[i_b].cpu().numpy().transpose(1, 2, 0) * 255).astype(
                    np.uint8
                )
                if len(det):
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (mix_train_set.COCO_CLASS[int(cls)], conf)
                        x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                        cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(cv2_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                result_imgs.append(cv2_img)
                # cv2.imwrite(os.path.join(sample_save_dir, f'detect-{idx}.png'), cv2_img)

            rows = cols = int(math.floor(math.log2(len(result_imgs))))
            new_num_imgs = rows * cols
            img_height, img_width = result_imgs[0].shape[: 2]
            concatenated_image = np.zeros((img_height * rows, img_width * cols, 3), dtype=np.uint8)
            for i in range(rows):
                for j in range(cols):
                    img_idx = i * cols + j
                    concatenated_image[i * img_height :(i + 1) * img_height,
                                       j * img_width :(j + 1) * img_width] = result_imgs[img_idx]
            cv2.imwrite(os.path.join(sample_save_dir, f'detect.png'), concatenated_image)
            # cv2.imwrite(os.path.join(sample_save_dir, f'detect-{idx}.png'), concatenated_image)
            detector.train()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        det_loss_val = loss_reduced["det"].mean().item()
        loss_box = loss_reduced["lbox"].mean().item()
        loss_obj = loss_reduced["lobj"].mean().item()
        loss_cls = loss_reduced["lcls"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        # path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        # path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"{cstr('epoch')}:{i} "
                    f"{cstr('D')}:{d_loss_val:.4f} "
                    f"{cstr('r1')}:{r1_val:.4f} "
                    f"{cstr('G')}:{g_loss_val:.4f} "
                    f"{cstr('det')}:{det_loss_val:.4f} "
                    f"{cstr('lbox')}:{loss_box:.4f} "
                    f"{cstr('lobj')}:{loss_obj:.4f} "
                    f"{cstr('lcls')}:{loss_cls:.4f} "

                                                         # f"{cstr('path')}:{path_loss_val:.4f} "
                                                         # f"{cstr('mpath')}:{mean_path_length_avg:.4f} "
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Detector": det_loss_val,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                                                        # "Path Length Regularization": path_loss_val,
                                                        # "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                                                        # "Path Length": path_length_val,
                    }
                )

            # if i % 100 == 0:
            #     with torch.no_grad():
            #         g_ema.eval()
            #         sample = g_ema(texture_latent_mix_noise,cond_latent)
            #         # utils.save_image(
            #         #     sample,
            #         #     f"{sample_save_dir}/{str(i).zfill(6)}.png",
            #         #     nrow=int(args.n_sample**0.5),
            #         #     normalize=True,
            #         #     range=(-1, 1),
            #         # )

            if i % 500 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    f"{checkpoint_save_dir}/{str(i).zfill(6)}.pt",
                )


def prepare_training(args):
    # ----------------------------------------------
    #   Load Data
    # ----------------------------------------------
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ]
    )

    mix_train_set = tsgan.data.CroppedCOCOCarlaMixDataset(
        'configs/dataset.yaml',
        is_train=False,                                    # TODO: 测试完后，修改为训练 True
        transform=transform,
        show_detail=True,
    )

    loader = data.DataLoader(
        mix_train_set,
        batch_size=args.batch,
        num_workers=args.num_workers,
        sampler=data_sampler(mix_train_set, shuffle=True, distributed=args.distributed),
        collate_fn=tsgan.data.CroppedCOCOCarlaMixDataset.collate_fn,
        drop_last=True,
    )
    # ----------------------------------------------
    #   Load Neural Renderer
    # ----------------------------------------------
    with open('data/models/selected_faces.txt', 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model,
        selected_faces=selected_faces,
        texture_size=args.num_feature,
        image_size=800,
        device=device,
    )
    neural_renderer.to(neural_renderer.textures.device)
    neural_renderer.renderer.to(neural_renderer.textures.device)
    # neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)
    print('get textures size:', neural_renderer.textures.size())

    # ----------------------------------------------
    #   Load Pretrained Autoencoder & Classifier
    # ----------------------------------------------
    encoder = tsgan.models.autoencoder.TextureEncoder(
        num_feature=args.num_feature,
        latent_dim=args.latent_dim,
    ).to(device)
    decoder = tsgan.models.autoencoder.TextureDecoder(
        latent_dim=args.latent_dim,
        num_points=args.npoint,
        num_feature=args.num_feature,
    ).to(device)
    autoencoder_pretrained = torch.load(args.autoencoder_pretrained, map_location="cpu")
    encoder.load_state_dict(autoencoder_pretrained['encoder'])
    decoder.load_state_dict(autoencoder_pretrained['decoder'])
    cond_model = resnet50()
    cond_model.fc = nn.Linear(
        cond_model.fc.in_features,
        mix_train_set.COCO_CATEGORIES_MAP.__len__(),
    )
    cond_model.load_state_dict(torch.load(args.classifier_pretrained, map_location="cpu"))
    cond_model.to(device)

    # with open(data, "r") as f:
    #     data_dict: dict = yaml.safe_load(f) # dictionary
    with open("configs/hyp.scratch-low.yaml", "r") as f:
        hyp: dict = yaml.safe_load(f)                                                            # load hyps dict
    nc = 80
    detector = Model("configs/yolov5s.yaml", ch=3, nc=nc, anchors=hyp.get('anchors')).to(device) # create
    detector.nc = nc                                                                             # attach number of classes to model
    detector.hyp = hyp                                                                           # attach hyperparameters to model
    compute_detector_loss = ComputeLoss(detector)
    ckpt = torch.load("pretrained/yolov5s.pt", map_location='cpu')
    csd = ckpt['model'].float().state_dict()                                                     # checkpoint state_dict as FP32
    detector.load_state_dict(csd, strict=False)                                                  # load
    detector.train()

    detector_eval = Model("configs/yolov5s.yaml", ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    detector_eval.nc = nc
    detector_eval.hyp = hyp
    ckpt = torch.load("pretrained/yolov5s.pt", map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    detector_eval.load_state_dict(csd, strict=False)
    detector_eval.eval()

    # ----------------------------------------------
    #   GAN
    # ----------------------------------------------
    generator = Generator(
        conditiom_latent_dim=cond_model.fc.in_features, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    generator.train()

    discriminator = Discriminator(args.latent_dim, cond_model.fc.in_features).to(device)
    discriminator.train()
    g_ema = Generator(
        style_dim=args.latent,
        conditiom_latent_dim=cond_model.fc.in_features,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.train()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0**g_reg_ratio, 0.99**g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0**d_reg_ratio, 0.99**d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        generator.load_state_dict(ckpt["g"], strict=False)
        discriminator.load_state_dict(ckpt["d"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    return (
        neural_renderer,
        encoder,
        decoder,
        cond_model,
        detector,
        detector_eval,
        compute_detector_loss,
        generator,
        discriminator,
        g_ema,
        g_optim,
        d_optim,
        loader,
        mix_train_set,
        device,
    )


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    return data.RandomSampler(dataset) if shuffle else data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def g_path_regularize(fake_latent, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_latent) / math.sqrt(fake_latent.size(1))
    grad, = autograd.grad(outputs=(fake_latent * noise).sum(), inputs=latents, create_graph=True)
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument('--save_dir', type=str, default='stylegan2')
    parser.add_argument("--iter", type=int, default=20000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=2, help="batch sizes for each gpus")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=64, help="number of the samples generated during training")
    parser.add_argument("--size", type=int, default=1024, help="image sizes for the model")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)"
    )
    parser.add_argument("--d_loss_every", type=int, default=32, help="interval of the r1 regularization")
    parser.add_argument("--d_reg_every", type=int, default=128, help="interval of the r1 regularization")
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the path length regularization")
    parser.add_argument("--g_det_every", type=int, default=1, help="interval of the path length regularization")
    parser.add_argument("--valid_every", type=int, default=100, help="interval of the path length regularization")
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1"
    )
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")

    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")

    parser.add_argument('--obj_model', type=str, default='data/models/vehicle-YZ.obj')
    parser.add_argument('--autoencoder_pretrained', type=str, default='tmp/autoencoder.pt')
    parser.add_argument('--classifier_pretrained', type=str, default='tmp/classifier/resnet50-4.pt')

    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    train(args)
