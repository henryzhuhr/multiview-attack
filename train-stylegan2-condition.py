import argparse
from copy import deepcopy
import logging
import math
from pathlib import Path
import random
import os
import sys
import cv2
import datetime

import numpy as np
import neural_renderer as nr
import torch
from torch import Tensor, nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, utils

from tqdm import tqdm
import yaml
import tsgan
from tsgan.render import NeuralRenderer

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
nowt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train(args):

    # sample_save_dir = os.path.join("tmp", args.save_dir, "sample")
    os.makedirs(sample_save_dir := os.path.join("tmp", args.save_dir, "sample"), exist_ok=True)
    # checkpoint_save_dir = os.path.join("tmp", args.save_dir, "checkpoint")
    os.makedirs(checkpoint_save_dir := os.path.join("tmp", args.save_dir, "checkpoint"), exist_ok=True)
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s] %(message)s',
        level=logging.INFO,
        filename=os.path.join("tmp", args.save_dir, f'train-{nowt}.log'),
        filemode='a',
    )

    (
        neural_renderer, encoder, decoder, cond_model, detector, detector_eval, compute_detector_loss, generator,
        discriminator, g_ema, g_optim, d_optim, scaler, train_loader, mix_train_set, device
    ) = prepare_training(args)

    with torch.no_grad():
        real_texture = neural_renderer.textures
        texture_latent = encoder.forward(real_texture)
        batch_texture_latent = texture_latent.repeat(args.batch, *[1] * (len(texture_latent.size()) - 1))

    # ----------------------------------------------
    #   start training
    # ----------------------------------------------
    for epoch in range(args.epochs):
        clr = g_optim.state_dict()['param_groups'][0]['lr']
        print(
            "\033[32m",
            f"[Epoch]{epoch}/{args.epochs}",
            f"[Batch Size]{args.batch}",
            f"[LR]{g_optim.state_dict()['param_groups'][0]['lr']:.6f}",
            f":{device}",
            "\033[0m",
        )

        g_loss_epoch = 0
        loss_d_epoch, loss_real_epoch, loss_fake_epoch = 0, 0, 0
        loss_r1_epoch = 0
        lbox_epoch, lobj_epoch, lcls_epoch = 0, 0, 0
        det_loss_epoch = 0
        data_num = 0

        accum = 0.5**(32 / (10 * 1000))

        pbar = tqdm(train_loader)
        for i_mini_batch, batch_data in enumerate(pbar):
            cond_images = batch_data["coco"]["image"].to(device)
            coco_label = batch_data["coco"]["predict_id"].to(device)
            carla_scene_names = batch_data["carla"]["name"]
            carla_scene_images = batch_data["carla"]["image"]
            carla_render_params = batch_data["carla"]["render_param"]
            BS = cond_images.shape[0]
            data_num += BS

            # ----------------------------------------
            # perform backward
            # ----------------------------------------
            texture_latent_mix_noise = batch_texture_latent.clone()
            # for i_b in range(BS):
            #     noise_rotio = 0.5
            #     noise = torch.randn_like(texture_latent_mix_noise[i_b])
            #     # texture_latent_mix_noise[i_b] = (1 - noise_rotio) * texture_latent_mix_noise[i_b] + noise_rotio * noise
            #     texture_latent_mix_noise[i_b] = texture_latent_mix_noise[i_b] 

            with torch.no_grad():
                cond_latent = cond_model.forward_latent(cond_images)


            # ----------------------------------------
            #  train Discriminator
            # ----------------------------------------
            requires_grad(generator, False)
            requires_grad(discriminator, True)
            with autocast():
                fake_texture_latent = generator.forward(texture_latent_mix_noise, cond_latent) # TODO: Inference

                real_pred = discriminator.forward(batch_texture_latent, cond_latent)
                fake_pred = discriminator.forward(fake_texture_latent.detach(), cond_latent)

                is_d_loss = i_mini_batch % args.d_loss_every == 0
                if is_d_loss:
                    d_real_loss = F.softplus(-real_pred).mean()
                    d_fake_loss = F.softplus(fake_pred).mean()

                    d_loss=d_real_loss+d_fake_loss# D logistic loss

                    loss_d_epoch += d_loss.item()
                    real_score = real_pred.mean().item()
                    loss_real_epoch += real_score
                    fake_score = fake_pred.mean().item()
                    loss_fake_epoch += fake_score

                fake_textures = decoder.forward(fake_texture_latent)
                render_image_list, render_scene_list, render_label_list = [], [], []

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

                    # scene_npimg: np.ndarray = t_render_image.detach().cpu().numpy() * 255
                    # print("name ",carla_scene_names[i_b])
                    # scene_npimg = scene_npimg.astype(np.uint8)
                    # scene_npimg = scene_npimg.transpose(1, 2, 0)
                    # scene_npimg = np.ascontiguousarray(scene_npimg)

                    render_image_list.append(t_rgb_image.unsqueeze(0))
                    render_scene_list.append(t_render_image.unsqueeze(0))

                    # find object label
                    ret, binary = cv2.threshold(
                        cv2.cvtColor(render_npimg, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY
                    )
                    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    find_boxes = []
                    for c in contours:
                        [x, y, w, h] = cv2.boundingRect(c)
                        find_boxes.append([x, y, x + w, y + h])
                    is_find_box = len(find_boxes) > 0
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

                pred = detector.forward(render_images)
                (det_loss, (lbox, lobj, lcls)) = compute_detector_loss.__call__(pred, render_labels)

                lbox_epoch += lbox.item() * BS
                lobj_epoch += lobj.item() * BS
                lcls_epoch += lcls.item() * BS
                det_loss_epoch += det_loss.item() * BS
                d_loss = det_loss + d_loss if is_d_loss else det_loss

            # D backward
            discriminator.zero_grad()
            d_loss.backward()
            clip_grad_norm_(parameters=discriminator.parameters(), max_norm=10, norm_type=2)
            d_optim.step()

            # ----------------------------------------
            #   D regularization for every d_reg_every iterations
            # ----------------------------------------
            r1_loss = 0
            if i_mini_batch % args.d_reg_every == 0:
                batch_texture_latent.requires_grad = True
                real_pred = discriminator(batch_texture_latent, cond_latent)
                r1_loss = d_r1_loss(real_pred, batch_texture_latent)

                loss_r1_epoch += r1_loss.item()* BS

                discriminator.zero_grad()
                scaler.scale(r1_loss).backward()
                # r1_loss.backward()
                scaler.step(d_optim)
                # d_optim.step()
                scaler.update()
                batch_texture_latent.requires_grad = False

            # frozen D
            requires_grad(generator, True)
            requires_grad(discriminator, False)

            # G backward
            # fake_texture = generator.forward(texture_latent_mix_noise, cond_latent)
            with autocast():
                fake_texture_latent = generator.forward(texture_latent_mix_noise, cond_latent) # TODO: Inference
                fake_pred = discriminator(fake_texture_latent, cond_latent)
                g_loss = g_nonsaturating_loss(fake_pred)
                g_loss_epoch += g_loss * BS

            generator.zero_grad()
            # g_loss.backward()
            scaler.scale(g_loss).backward()
            # g_optim.step()
            clip_grad_norm_(parameters=generator.parameters(), max_norm=10, norm_type=2)
            scaler.step(g_optim)
            scaler.update()

            # ----------------------------------------
            #   G regularization for every g_reg_every iterations
            # ----------------------------------------
            # if i_mini_batch % args.g_reg_every == 0:
            #     fake_texture_latent, latents = generator.forward(
            #         texture_latent_mix_noise, cond_latent, return_latents=True
            #     )
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
            #     loss_dict["path"] = path_loss
            #     loss_dict["path_length"] = path_lengths.mean()

            accumulate(g_ema, generator, accum)
            pbar.set_description(" ".join((
                f"{cstr('G')}:{g_loss.item():.4f}",
                f"{cstr('D')}:{d_loss.item():.4f}",
                f"{cstr('det')}:{det_loss.item():.4f}",
                f"{cstr('lbox')}:{lbox.item():.4f}",
                f"{cstr('lobj')}:{lobj.item():.4f}",
                f"{cstr('lcls')}:{lcls.item():.4f}",
                f"{cstr('Rs')}:{real_score:.4f}" if is_d_loss else "",
                f"{cstr('Fs')}:{fake_score:.4f}" if is_d_loss else "",
                f"{cstr('R1')}:{r1_loss.item():.4f}" if (i_mini_batch % args.d_reg_every == 0) else "",
            )))# yapf:disable
        # ----------------------------------------
        #  Valid
        # ----------------------------------------
        if (render_scenes is not None):

            with torch.no_grad():
                pred = detector_eval.forward(render_scenes)

                conf_thres, iou_thres = 0.25, 0.6
                pred = non_max_suppression(pred, conf_thres, iou_thres, None, False)
                result_imgs = []
                for i_b, det in enumerate(pred):
                    cv2_img = np.ascontiguousarray(render_scenes[i_b].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)#yapf:disable
                    if len(det):
                        for *xyxy, conf, cls in det:
                            label = '%s %.2f' % (mix_train_set.COCO_CLASS[int(cls)], conf)
                            x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                            cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(cv2_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    result_imgs.append(cv2_img)

                rows = cols = int(math.log2(len(result_imgs)))
                new_num_imgs = rows * cols
                img_height, img_width = result_imgs[0].shape[: 2]
                concatenated_image = np.zeros((img_height * rows, img_width * cols, 3), dtype=np.uint8)
                for i in range(rows):
                    for j in range(cols):
                        img_idx = i * cols + j
                        if img_idx < len(result_imgs):
                            concatenated_image[i * img_height :(i + 1) * img_height,
                                               j * img_width :(j + 1) * img_width] = result_imgs[img_idx]
                cv2.imwrite(os.path.join(sample_save_dir, f'detect.png'), concatenated_image)
                cv2.imwrite(os.path.join(sample_save_dir, f'detect-{epoch}_{i_mini_batch}.png'), concatenated_image)

        epoch_loss_dict = {
            "G": g_loss_epoch / data_num,
            "D": loss_d_epoch / data_num * args.d_loss_every,
            "Det": det_loss_epoch / data_num,
            "lbox": lbox_epoch / data_num,
            "lobj": lobj_epoch / data_num,
            "lcls": lcls_epoch / data_num,
            "Rs": loss_real_epoch / data_num * args.d_loss_every,
            "Fs": loss_fake_epoch / data_num * args.d_loss_every,
            "R1": loss_r1_epoch / data_num * args.d_reg_every,
        }

        log_info = " ".join(["epoch:%-4d" % epoch] + [f"{k}:{v:.5f}" for k, v in epoch_loss_dict.items()])
        logging.info(log_info)
        print(log_info)
        print()

        if i % args.valid_every == 0:
            torch.save(
                {
                    "g": generator.state_dict(),
                                                               # "d": d_module.state_dict(),
                                                               # "g_ema": g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                                                               # "d_optim": d_optim.state_dict(),
                    "args": args,
                },
                f"{checkpoint_save_dir}/{str(i).zfill(6)}.pt",
            )


def prepare_training(args):
    # ----------------------------------------------
    #   Load Data
    # ----------------------------------------------

    mix_train_set = tsgan.data.CroppedCOCOCarlaMixDataset(
        'configs/dataset.yaml',
        is_train=True,                                    # TODO: 测试完后, False 修改为训练 True
        show_detail=True,
        # load_all_class=True,
        transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
        ]),
    )# yapf:disable

    if False:
        os.makedirs(tmp_dataset_image_save := "tmp/mix_coco", exist_ok=True)
        for i_ds, data_dict in enumerate(mix_train_set):
            coco_sample = data_dict["coco"]
            cv2.imwrite(
                f"{tmp_dataset_image_save}/{i_ds}-{coco_sample['category_name']}.png",
                (coco_sample["image"].numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
            )

    train_loader = data.DataLoader(
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
    ).cuda().train()
    decoder = tsgan.models.autoencoder.TextureDecoder(
        latent_dim=args.latent_dim,
        num_points=args.npoint,
        num_feature=args.num_feature,
    ).cuda().train()
    # autoencoder_pretrained = torch.load(args.autoencoder_pretrained, map_location="cpu")
    # encoder.load_state_dict(autoencoder_pretrained['encoder'])
    # decoder.load_state_dict(autoencoder_pretrained['decoder'])
    cond_model = resnet50()
    cond_model.fc = nn.Linear(
        cond_model.fc.in_features,
        mix_train_set.COCO_CATEGORIES_MAP.__len__(),
    )
    cond_model.load_state_dict(torch.load(args.classifier_pretrained, map_location="cpu"))
    cond_model.cuda().eval()

    # ----------------------------------------------
    #   Detector
    # ----------------------------------------------
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

    detector_eval = deepcopy(detector).eval() # for inference

    # ----------------------------------------------
    #   GAN
    # ----------------------------------------------
    generator = Generator(
        style_dim=args.latent_dim,
        conditiom_latent_dim=cond_model.fc.in_features,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        mix_prob=args.mix_prob,
    ).cuda()
    generator.train()
    g_ema = Generator(
        style_dim=args.latent_dim,
        conditiom_latent_dim=cond_model.fc.in_features,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        mix_prob=args.mix_prob,
    ).cuda()
    g_ema.train()
    discriminator = Discriminator(args.latent_dim, cond_model.fc.in_features).to(device)
    discriminator.train()

    scaler = GradScaler() # mixed precision training
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        [{
            "params": generator.parameters()
        }, {
            "params": encoder.parameters()
        }, {
            "params": decoder.parameters()
        }],
        lr=args.lr * g_reg_ratio,
        betas=(0**g_reg_ratio, 0.99**g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0**d_reg_ratio, 0.99**d_reg_ratio),
    )

    # if args.ckpt is not None:
    #     print("load model:", args.ckpt)

    #     ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

    #     generator.load_state_dict(ckpt["g"], strict=False)
    #     discriminator.load_state_dict(ckpt["d"], strict=False)
    #     g_ema.load_state_dict(ckpt["g_ema"], strict=False)

    #     g_optim.load_state_dict(ckpt["g_optim"])
    #     d_optim.load_state_dict(ckpt["d_optim"])

    return (
        neural_renderer, encoder, decoder, cond_model, detector, detector_eval, compute_detector_loss, generator,
        discriminator, g_ema, g_optim, d_optim, scaler, train_loader, mix_train_set, device
    )


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    return data.RandomSampler(dataset) if shuffle else data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1: nn.Module, model2: nn.Module, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred: Tensor) -> Tensor:
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
    parser.add_argument("--epochs", type=int, default=20000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=2, help="batch sizes for each gpus")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--size", type=int, default=1024, help="feature size for G")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
    parser.add_argument("--path_batch_shrink", type=int, default=2, help="batch reduce factor (reduce memory use)")

    parser.add_argument("--d_loss_every", type=int, default=2, help="interval of the r1 regularization")
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the r1 regularization")
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the path length regularization")
    parser.add_argument("--g_det_every", type=int, default=1, help="interval of the path length regularization")
    parser.add_argument("--valid_every", type=int, default=100, help="interval of the path length regularization")
    parser.add_argument("--mix_prob", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor")

    parser.add_argument('--obj_model', type=str, default='data/models/vehicle-YZ.obj')
    parser.add_argument('--npoint', type=int, default=12306)
    parser.add_argument('--num_feature', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=2048)
    parser.add_argument('--autoencoder_pretrained', type=str, default='tmp/autoencoder/autoencoder.pt')
    parser.add_argument('--classifier_pretrained', type=str, default='tmp/classifier/resnet50-4.pt')

    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.n_mlp = 8

    train(args)
