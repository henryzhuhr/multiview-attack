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

from models.gan import TextureGenerator
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


class ArgsType:
    save_dir: str
    epochs: int
    batch: int
    num_workers: int
    size: int

    mix_prob: float
    lr: float

    obj_model: str
    selected_faces: str
    texture_size: int
    latent_dim: int
    pretrained: str


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='stylegan2')
    parser.add_argument("--epochs", type=int, default=20000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=8, help="batch sizes for each gpus")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--size", type=int, default=1024, help="feature size for G")

    parser.add_argument("--mix_prob", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")

    parser.add_argument('--obj_model', type=str, default="data/models/vehicle-YZ.obj")
    parser.add_argument('--selected_faces', type=str, default="data/models/selected_faces.txt")
    parser.add_argument('--texture_size', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--pretrained', type=str)

    return parser.parse_args()


def prepare_training(args: ArgsType):
    # ----------------------------------------------
    #   Load Data
    # ----------------------------------------------
    train_set = tsgan.data.CroppedCOCOCarlaMixDataset(
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

    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch,
        num_workers=args.num_workers,
        collate_fn=tsgan.data.CroppedCOCOCarlaMixDataset.collate_fn,
        drop_last=True,
    )
    # --- Load Neural Renderer ---
    with open(args.selected_faces, 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model,
        selected_faces=selected_faces,
        texture_size=args.texture_size,
        image_size=800,
        device="cuda",
    )
    x_t = neural_renderer.textures[:, selected_faces, :]
    nt = x_t.shape[1]
    print('textures num:%d (%d selected)' % (neural_renderer.textures.shape[1], nt))

    # ----------------------------------------------
    #   Detector
    # ----------------------------------------------
    with open("configs/hyp.scratch-low.yaml", "r") as f:
        hyp: dict = yaml.safe_load(f)                                                        # load hyps dict
    nc = 80
    detector = Model("configs/yolov5s.yaml", ch=3, nc=nc, anchors=hyp.get('anchors')).cuda() # create
    detector.nc = nc                                                                         # attach number of classes to model
    detector.hyp = hyp                                                                       # attach hyperparameters to model
    detector_loss = ComputeLoss(detector)
    ckpt = torch.load("pretrained/yolov5s.pt", map_location='cpu')
    csd = ckpt['model'].float().state_dict()                                                 # checkpoint state_dict as FP32
    detector.load_state_dict(csd, strict=False)                                              # load
    detector.eval()

    # ----------------------------------------------
    #   GAN
    # ----------------------------------------------
    model = TextureGenerator(
        nt=nt,
        ts=args.texture_size,
        style_dim=args.latent_dim,
        cond_dim=len(train_set.COCO_CLASS),
        mix_prob=args.mix_prob,
    )
    model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.cuda().train()
    model.encoder.eval()
    # model.decoder.eval()

    optimized_params = [{"params": model.generator.parameters()}, {"params": model.decoder.parameters()}]
    # optimizer = optim.Adam(optimized_params, lr=args.lr,weight_decay=1e-4)
    optimizer = optim.SGD(optimized_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    lr_heduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.1)

    return (neural_renderer, model, detector, detector_loss, optimizer, lr_heduler, train_loader, train_set)


def train():
    args: ArgsType = get_args()
    nowt = datetime.datetime.now().strftime("%m%d%H%M")
    args.save_dir = args.save_dir # + "-" + nowt

    # sample_save_dir = os.path.join("tmp", args.save_dir, "sample")
    os.makedirs(sample_save_dir := os.path.join("tmp", args.save_dir, "sample"), exist_ok=True)
    # checkpoint_save_dir = os.path.join("tmp", args.save_dir, "checkpoint")
    os.makedirs(checkpoint_save_dir := os.path.join("tmp", args.save_dir, "checkpoint"), exist_ok=True)
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s] %(message)s',
        level=logging.INFO,
        filename=os.path.join("tmp", args.save_dir, f'train.log'),
        filemode='a',
    )

    (neural_renderer, model, detector, detector_loss, optimizer, lr_heduler, train_loader,
     mix_train_set) = prepare_training(args)
    device = "cuda"
    tt = neural_renderer.textures[:, neural_renderer.selected_faces, :]
    real_x = tt.repeat(args.batch, *[1] * (len(tt.size()) - 1))
    # n_r = 0.3
    # real_x = (1 - n_r) * real_x + n_r * torch.rand_like(real_x)

    # ----------------------------------------------
    #   start training
    # ----------------------------------------------
    for epoch in range(args.epochs):
        clr = optimizer.state_dict()['param_groups'][0]['lr']
        print(
            "\033[32m",
            f"[Epoch]{epoch}/{args.epochs}",
            f"[Batch Size]{args.batch}",
            f"[LR]{clr:.6f}",
            f":{device}",
            "\033[0m",
        )

        loss_smooth_epoch = 0
        ldet_epoch, lbox_epoch, lobj_epoch, lcls_epoch = 0, 0, 0, 0
        ladv_epoch = 0
        data_num = 0

        accum = 0.5**(32 / (10 * 1000))

        pbar = tqdm(train_loader)
        model.generator.train()
        if True:
            for i_mb, batch_data in enumerate(pbar):
                cond_images = batch_data["coco"]["image"].to(device)
                coco_label = batch_data["coco"]["predict_id"].to(device)
                carla_scene_images = batch_data["carla"]["image"]

                carla_render_params = batch_data["carla"]["render_param"]
                bs = cond_images.shape[0]
                data_num += bs

                fake_xs = model.decode(model.forward(real_x, coco_label))

                _t = neural_renderer.textures.repeat(
                    fake_xs.shape[0], *[1] * (len(neural_renderer.textures.size()) - 1)
                )
                _t[:, neural_renderer.selected_faces, :] = fake_xs
                fake_textures = _t

                render_image_list, render_scene_list, render_label_list = [], [], []
                for i_b in range(fake_textures.size(0)):
                    carla_scene_image = carla_scene_images[i_b]
                    t_scene_image = torch.from_numpy(carla_scene_image).to(fake_textures.device).permute(2, 0, 1).float() / 255.  # yapf: disable

                    crp = carla_render_params[i_b] # carla_render_param
                    fake_texture = fake_textures[i_b].unsqueeze(0)

                    neural_renderer.set_render_perspective(crp["camera_transform"], crp["vehicle_transform"],crp["fov"])        # yapf: disable
                    (f_rgb, _, f_alpha) = neural_renderer.renderer.forward(
                        neural_renderer.vertices, neural_renderer.faces, F.tanh(fake_texture)
                    )

                    f_render_image = f_alpha * f_rgb[0] + (1 - f_alpha) * t_scene_image

                    rgb_img: np.ndarray = (f_rgb[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)   # yapf: disable

                    # t_rgb_image = rgb_images.squeeze(0)
                    # t_alpha_image = alpha_images
                    # t_scene_image = torch.from_numpy(carla_scene_image).to(t_rgb_image.device).permute(2, 0, 1).float() / 255.  # yapf: disable
                    render_image = f_alpha * f_rgb.squeeze(0) + (1 - f_alpha) * t_scene_image
                    t_scene_image = torch.from_numpy(carla_scene_image).to(f_rgb.device).permute(2, 0, 1).float() / 255.  # yapf: disable

                    scene_img: np.ndarray = np.ascontiguousarray(render_image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)   # yapf: disable

                    render_image_list.append(f_rgb)
                    render_scene_list.append(render_image)

                    # find object label
                    binary=np.ascontiguousarray(f_alpha.squeeze(0).detach().cpu().numpy()*255).astype(np.uint8)
                    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    find_boxes = []
                    for c in contours:
                        [x, y, w, h] = cv2.boundingRect(c)
                        find_boxes.append([x, y, x + w, y + h])
                    fc = np.array(find_boxes)

                    box = [min(fc[:, 0]), min(fc[:, 1]), max(fc[:, 2]), max(fc[:, 3])] # [x1,y1,x2,y2]

                    # if True:
                    #     [x1, y1, x2, y2] = box
                    #     cv2.rectangle(scene_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #     label = mix_train_set.COCO_CLASS[int(coco_label[i_b])]
                    #     cv2.putText(scene_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #     cv2.imwrite('tmp/render.png', scene_img)
                    # exit()

                    b,c, h, w = f_rgb.shape
                    render_label_list.append(torch.tensor([
                        i_b, int(coco_label[i_b]),
                        (box[0] + box[2]) / 2 / w,
                        (box[1] + box[3]) / 2 / h,
                        (box[2] - box[0]) / w,
                        (box[3] - box[1]) / h,
                    ])) # yapf:disable
                render_images = torch.stack(render_image_list, dim=0).to(device)
                render_scenes = torch.stack(render_scene_list, dim=0).to(device)
                render_labels = torch.stack(render_label_list, dim=0).to(device)

                _, pred = detector.forward(render_scenes)
                (det_loss, (lbox, lobj, lcls)) = detector_loss.__call__(pred, render_labels)
                lbox *= 0.05
                lobj *= 1.0
                lcls *= 0.5 * 1.5
                lbox_epoch += lbox.item() * bs
                lobj_epoch += lobj.item() * bs
                lcls_epoch += lcls.item() * bs
                ldet_epoch += det_loss.item() * bs



                if i_mb%4==0:
                    loss_smooth = 0.05 * F.l1_loss(real_x, fake_xs)
                    loss_smooth_epoch += loss_smooth.item() * bs/4
                else:
                    loss_smooth=0

                loss_adv = lbox + lobj + lcls+loss_smooth

                ladv_epoch += loss_adv.item() * bs

                optimizer.zero_grad()
                loss_adv.backward()
                # clip_grad_norm_(parameters=g_model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()

                # accumulate(g_ema, generator, accum)

                pbar.set_description(" ".join((
                    f"{cstr('adv')}:{loss_adv.item():.4f}",
                    # f"{cstr('Ls')}:{loss_smooth.item():.4f}",
                    f"{cstr('det')}:{det_loss.item():.4f}",
                    f"{cstr('lcls')}:{lcls.item():.4f}",
                    f"{cstr('lbox')}:{lbox.item():.4f}",
                    f"{cstr('lobj')}:{lobj.item():.4f}",
                    f"{cstr('label')}:{int(coco_label[0])}",
                )))# yapf:disable
        lr_heduler.step()

        #  Valid
        if True:
            model.generator.eval()

            for batch_data in train_loader:
                cond_images = batch_data["coco"]["image"].to(device)
                coco_label = batch_data["coco"]["predict_id"].to(device)
                carla_scene_images = batch_data["carla"]["image"]
                carla_render_params = batch_data["carla"]["render_param"]
                bs = cond_images.shape[0]
                break

            with torch.no_grad():

                tt = neural_renderer.textures[:, neural_renderer.selected_faces, :]
                real_x = tt.repeat(args.batch, *[1] * (len(tt.size()) - 1))
                # n_r = 0.3
                # real_x = (1 - n_r) * real_x + n_r * torch.rand_like(real_x)

                fake_xs = model.decode(model.forward(real_x, coco_label))

                tt = neural_renderer.textures
                _t = tt.repeat(fake_xs.shape[0], *[1] * (len(tt.shape) - 1))
                _t[:, neural_renderer.selected_faces, :] = fake_xs
                fake_textures = _t

                _t = tt.repeat(real_x.shape[0], *[1] * (len(tt.shape) - 1))
                _t[:, neural_renderer.selected_faces, :] = real_x
                real_textures = _t

                _t = tt.repeat(real_x.shape[0], *[1] * (len(tt.shape) - 1))
                _t[:, neural_renderer.selected_faces, :] = torch.rand_like(real_x).to(real_x.device)
                noise_textures = _t

                real_image_list, fake_image_list, noise_images_list = [], [], []
                for i_b in range(fake_textures.size(0)):
                    carla_scene_image = carla_scene_images[i_b]
                    t_scene_image = torch.from_numpy(carla_scene_image).to(fake_textures.device).permute(2, 0, 1).float() / 255.  # yapf: disable

                    crp = carla_render_params[i_b] # carla_render_param
                    fake_texture = fake_textures[i_b].unsqueeze(0)

                    neural_renderer.set_render_perspective(crp["camera_transform"], crp["vehicle_transform"],crp["fov"])        # yapf: disable
                    (r_rgb, _, r_alpha) = neural_renderer.renderer.forward(
                        neural_renderer.vertices, neural_renderer.faces, F.tanh(real_textures)
                    )
                    r_render_image = r_alpha * r_rgb[0] + (1 - r_alpha) * t_scene_image
                    real_image_list.append(r_render_image.unsqueeze(0))

                    (f_rgb, _, f_alpha) = neural_renderer.renderer.forward(
                        neural_renderer.vertices, neural_renderer.faces, F.tanh(fake_texture)
                    )
                    f_render_image = f_alpha * f_rgb[0] + (1 - f_alpha) * t_scene_image
                    fake_image_list.append(f_render_image.unsqueeze(0))

                    (n_rgb, _, n_alpha) = neural_renderer.renderer.forward(
                        neural_renderer.vertices, neural_renderer.faces, F.tanh(noise_textures)
                    )
                    n_render_image = n_alpha * n_rgb[0] + (1 - n_alpha) * t_scene_image
                    noise_images_list.append(n_render_image.unsqueeze(0))
                real_images = torch.cat(real_image_list, dim=0).to(device)
                fake_images = torch.cat(fake_image_list, dim=0).to(device)
                noise_images = torch.cat(noise_images_list, dim=0).to(device)

                with torch.no_grad():
                    r_pred = detector.forward(real_images)  # real
                    f_pred = detector.forward(fake_images)  # fake
                    n_pred = detector.forward(noise_images) # noise

                conf_thres, iou_thres = 0.25, 0.6

                r_result_imgs = []
                for i_b, det in enumerate(non_max_suppression(r_pred, conf_thres, iou_thres, None, False)):
                    cv_img = np.ascontiguousarray(real_images[i_b].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)#yapf:disable
                    if len(det):
                        for *xyxy, conf, cls in det:
                            label = '%s %.2f' % (mix_train_set.COCO_CLASS[int(cls)], conf)
                            x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(cv_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    r_result_imgs.append(cv_img)

                f_result_imgs = []
                for i_b, det in enumerate(non_max_suppression(f_pred, conf_thres, iou_thres, None, False)):
                    cv_img = np.ascontiguousarray(fake_images[i_b].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)#yapf:disable
                    if len(det):
                        for *xyxy, conf, cls in det:
                            label = '%s %.2f' % (mix_train_set.COCO_CLASS[int(cls)], conf)
                            x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(cv_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    f_result_imgs.append(cv_img)

                n_result_imgs = []
                for i_b, det in enumerate(non_max_suppression(n_pred, conf_thres, iou_thres, None, False)):
                    cv_img = np.ascontiguousarray(noise_images[i_b].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)#yapf:disable
                    if len(det):
                        for *xyxy, conf, cls in det:
                            label = '%s %.2f' % (mix_train_set.COCO_CLASS[int(cls)], conf)
                            x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(cv_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    n_result_imgs.append(cv_img)

                result_imgs = [
                    cv2.hconcat([r_result_imgs[i], f_result_imgs[i], n_result_imgs[i]])
                    for i in range(len(r_result_imgs))
                ]

                rows = cols = int(math.log2(len(result_imgs)))
                img_height, img_width = result_imgs[0].shape[: 2]
                concatenated_image = np.zeros((img_height * rows, img_width * cols, 3), dtype=np.uint8)
                for i in range(rows):
                    for j in range(cols):
                        img_idx = i * cols + j
                        if img_idx < len(result_imgs):
                            concatenated_image[i * img_height :(i + 1) * img_height,
                                               j * img_width :(j + 1) * img_width] = result_imgs[img_idx]
                cv2.imwrite(os.path.join(sample_save_dir, f'_detect.png'), concatenated_image)
                cv2.imwrite(os.path.join(sample_save_dir, f'detect-{epoch}.png'), concatenated_image)
                # [
                #     cv2.imwrite(os.path.join(sample_save_dir, f'detect-{epoch}_{i_img}.png'), result_imgs[i])
                #     for i_img in range(len(result_imgs))
                # ]

        epoch_loss_dict = {
            "Adv": (ladv_epoch) / data_num,
            "Det": ldet_epoch / data_num,
            "Ls": loss_smooth_epoch / data_num/4,
            "lcls": lcls_epoch / data_num,
            "lbox": lbox_epoch / data_num,
            "lobj": lobj_epoch / data_num,
            "lr": clr,
        }

        logging.info(" ".join(["epoch:%-4d" % epoch] + [f"{k}:{v:.5f}" for k, v in epoch_loss_dict.items()]))
        print(" ".join([f"\033[00;34m{k}\033[0m:{v:.5f}" for k, v in epoch_loss_dict.items()]))
        print()

        torch.save(
            {
                "g": model.state_dict(),
                "g_optim": optimizer.state_dict(),
                "args": args
            }, f"{checkpoint_save_dir}/gan-{epoch}.pt"
        )


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    return data.RandomSampler(dataset) if shuffle else data.SequentialSampler(dataset)


def accumulate(model1: nn.Module, model2: nn.Module, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)



if __name__ == "__main__":
    train()
