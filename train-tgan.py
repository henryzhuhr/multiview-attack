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

from models.gan import TextureGenerator, Discriminator
import tsgan
from tsgan.render import NeuralRenderer

from tsgan.models.op import conv2d_gradfix
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
logt = lambda: "\033[01;32m{%d}\033[0m" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train(args):

    # sample_save_dir = os.path.join("tmp", args.save_dir, "sample")
    os.makedirs(sample_save_dir := os.path.join("tmp", args.save_dir, "sample"), exist_ok=True)
    # checkpoint_save_dir = os.path.join("tmp", args.save_dir, "checkpoint")
    os.makedirs(checkpoint_save_dir := os.path.join("tmp", args.save_dir, "checkpoint"), exist_ok=True)
    nowt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s] %(message)s',
        level=logging.INFO,
        filename=os.path.join("tmp", args.save_dir, f'train-{nowt}.log'),
        filemode='a',
    )

    (
        neural_renderer, g_model, d_model, detector, detector_eval, compute_detector_loss, g_optim, d_optim,
        train_loader, mix_train_set, device
    ) = prepare_training(args)

    with torch.no_grad():
        tt = neural_renderer.textures[:, neural_renderer.selected_faces, :]
        real_x = tt.repeat(args.batch, *[1] * (len(tt.size()) - 1)).detach().clone()
        real_x = torch.randn_like(real_x).to(real_x.device)

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
        loss_d_epoch, loss_smooth_epoch = 0, 0
        loss_r1_epoch = 0
        lbox_epoch, lobj_epoch, lcls_epoch = 0, 0, 0
        det_loss_epoch = 0
        data_num = 0

        accum = 0.5**(32 / (10 * 1000))

        pbar = tqdm(train_loader)
        g_model.train()
        d_model.train()
        for i_mini_batch, batch_data in enumerate(pbar):
            cond_images = batch_data["coco"]["image"].to(device)
            coco_label = batch_data["coco"]["predict_id"].to(device)
            carla_scene_images = batch_data["carla"]["image"]
            carla_render_params = batch_data["carla"]["render_param"]
            bs = cond_images.shape[0]
            data_num += bs

            # ----------------------------------------
            # perform backward
            # ----------------------------------------
            noise_rotio = 0.7
            random_z = torch.randn_like(real_x).to(real_x.device)
            # random_z = noise_rotio * random_z + (1 - noise_rotio) * real_x # add
            # ri = random.sample(range(real_x.shape[1]), int(real_x.shape[1] * (1 - noise_rotio)))
            # random_z[: ,ri] = real_x[: ,ri] # mask

            # ----------------------------------------
            #  train Discriminator
            # ----------------------------------------
            fake_latent = g_model.forward(random_z, coco_label)
            real_pred = d_model.forward(g_model.encode(real_x))
            fake_pred = d_model.forward(fake_latent)

            requires_grad(g_model, False)
            requires_grad(d_model, True)
            is_d_loss = i_mini_batch % args.d_loss_every == 0
            if is_d_loss:
                # if False:
                d_real_loss = F.softplus(-real_pred).mean()
                d_fake_loss = F.softplus(fake_pred).mean()

                d_loss = (d_real_loss + d_fake_loss)
                loss_d_epoch += d_loss.item()

                d_model.zero_grad()
                d_loss.backward()
                clip_grad_norm_(parameters=d_model.parameters(), max_norm=10, norm_type=2)
                d_optim.step()

            # ----------------------------------------
            #   D regularization for every d_reg_every iterations
            # ----------------------------------------
            r1_loss = 0
            if i_mini_batch % args.d_reg_every == 0:
                # if False == 0:
                real_x_latent = g_model.encode(real_x)
                real_x_latent.requires_grad = True
                real_pred = d_model.forward(real_x_latent)
                r1_loss = d_r1_loss(real_pred, real_x_latent)

                loss_r1_epoch += r1_loss.item() * bs

                d_model.zero_grad()
                r1_loss.backward()
                d_optim.step()

            # ----------------------------------------
            #  train generator: frozen D
            # ----------------------------------------
            requires_grad(g_model, True)
            requires_grad(d_model, False)
            if True:
                fake_xs = g_model.decode(g_model.forward(random_z, coco_label))
                _t = neural_renderer.textures.repeat(
                    fake_xs.shape[0], *[1] * (len(neural_renderer.textures.size()) - 1)
                )
                _t[:, neural_renderer.selected_faces, :] = fake_xs
                fake_textures = _t

                render_image_list, render_scene_list, render_label_list = [], [], []

                for i_b in range(fake_textures.size(0)):
                    carla_scene_image = carla_scene_images[i_b]
                    crp = carla_render_params[i_b] # carla_render_param
                    fake_texture = fake_textures[i_b].unsqueeze(0)

                    neural_renderer.set_render_perspective(crp["camera_transform"], crp["vehicle_transform"],crp["fov"])        # yapf: disable
                    (rgb_images, depth_images, alpha_images) = neural_renderer.renderer.forward(
                        neural_renderer.vertices, neural_renderer.faces, torch.tanh(fake_texture)
                    )

                    t_rgb_image = rgb_images.squeeze(0)
                    t_alpha_image = alpha_images
                    t_scene_image = torch.from_numpy(carla_scene_image).to(t_rgb_image.device).permute(2, 0, 1).float() / 255.  # yapf: disable
                    t_render_image = t_alpha_image * t_rgb_image + (1 - t_alpha_image) * t_scene_image

                    render_npimg: np.ndarray = (t_rgb_image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)   # yapf: disable

                    render_image_list.append(t_rgb_image.unsqueeze(0))
                    render_scene_list.append(t_render_image.unsqueeze(0))

                    # find object label
                    ret, binary = cv2.threshold(cv2.cvtColor(render_npimg, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)   # yapf: disable
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

                lbox_epoch += lbox.item() * bs
                lobj_epoch += lobj.item() * bs
                lcls_epoch += lcls.item() * bs
                det_loss_epoch += det_loss.item() * bs

            g_model.zero_grad()
            det_loss.backward()
            clip_grad_norm_(parameters=g_model.parameters(), max_norm=10, norm_type=2)
            g_optim.step()

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

            # accumulate(g_ema, generator, accum)

            pbar.set_description(" ".join((
                f"{cstr('D')}:{d_loss.item():.4f}",
                f"{cstr('det')}:{det_loss.item():.4f}",
                f"{cstr('lbox')}:{lbox.item():.4f}",
                f"{cstr('lobj')}:{lobj.item():.4f}",
                f"{cstr('lcls')}:{lcls.item():.4f}",
                f"{cstr('Ls')}:{loss_smooth_epoch:.4f}",
            )))# yapf:disable



        #  Valid
        if True:
            g_model.eval()            
            with torch.no_grad():
                fake_xs = g_model.decode(g_model.forward(random_z, coco_label))

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
                cv2.imwrite(os.path.join(sample_save_dir, f'detect-{epoch}.png'), concatenated_image)

        epoch_loss_dict = {
            "D": loss_d_epoch / data_num * args.d_loss_every,
            "Det": det_loss_epoch / data_num,
            "lbox": lbox_epoch / data_num,
            "lobj": lobj_epoch / data_num,
            "lcls": lcls_epoch / data_num,
            "Ls": loss_smooth_epoch / data_num * args.d_loss_every,
        }

        logging.info(" ".join(["epoch:%-4d" % epoch] + [f"{k}:{v:.5f}" for k, v in epoch_loss_dict.items()]))
        print(" ".join([f"\033[00;34m{k}\033[0m:{v:.5f}" for k, v in epoch_loss_dict.items()]))
        print()

        if i % args.valid_every == 0:
            torch.save(
                {
                    "g": g_model.state_dict(),
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
        is_train=False,                                    # TODO: 测试完后, False 修改为训练 True
        show_detail=True,
        # load_all_class=True,
        transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
        ]),
    )   # yapf:disable

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
    with open(args.selected_faces, 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model,
        selected_faces=selected_faces,
        texture_size=args.texture_size,
        image_size=800,
        device=device,
    )
    neural_renderer.to(neural_renderer.textures.device)
    neural_renderer.renderer.to(neural_renderer.textures.device)
    x_t = neural_renderer.textures[:, selected_faces, :]
    npoint = x_t.shape[1]
    print('textures num:%d (%d selected)' % (neural_renderer.textures.shape[1], npoint))

    # ----------------------------------------------
    #   Detector
    # ----------------------------------------------
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
    g_model = TextureGenerator(
        npoint=npoint,
        ts=args.texture_size,
        style_dim=args.latent_dim,
        cond_dim=len(mix_train_set.COCO_CLASS),
        mix_prob=args.mix_prob,
    ).cuda().train()
    d_model = Discriminator(latent_dim=args.latent_dim, cond_dim=len(mix_train_set.COCO_CLASS)).cuda().train()

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        g_model.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0**g_reg_ratio, 0.99**g_reg_ratio),
    )
    d_optim = optim.Adam(
        d_model.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0**d_reg_ratio, 0.99**d_reg_ratio),
    )

    return (
        neural_renderer, g_model, d_model, detector, detector_eval, compute_detector_loss, g_optim, d_optim,
        train_loader, mix_train_set, device
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


def g_path_regulzarize(fake_latent, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_latent) / math.sqrt(fake_latent.size(1))
    grad, = autograd.grad(outputs=(fake_latent * noise).sum(), inputs=latents, create_graph=True)
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument('--save_dir', type=str, default='stylegan2')
    parser.add_argument("--epochs", type=int, default=20000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=8, help="batch sizes for each gpus")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--size", type=int, default=1024, help="feature size for G")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
    parser.add_argument("--path_batch_shrink", type=int, default=2, help="batch reduce factor (reduce memory use)")

    parser.add_argument("--d_loss_every", type=int, default=2, help="interval of the r1 regularization")
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the r1 regularization")
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the path length regularization")
    parser.add_argument("--valid_every", type=int, default=100, help="interval of the path length regularization")
    parser.add_argument("--mix_prob", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor")

    parser.add_argument('--obj_model', type=str, default="data/models/vehicle-YZ.obj")
    parser.add_argument('--selected_faces', type=str, default="data/models/selected_faces.txt")
    parser.add_argument('--texture_size', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--classifier_pretrained', type=str, default='tmp/classifier/resnet50-4.pt')

    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    args.n_mlp = 8

    train(args)
