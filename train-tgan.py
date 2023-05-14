import os, sys
import argparse
import logging

from pathlib import Path
import random

from typing import List
import cv2
import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data

import tqdm
import yaml

from models.gan import TextureGenerator
from models.render import NeuralRenderer
from models.data.carladataset import CarlaDataset
from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import non_max_suppression

cstr = lambda s: f"\033[01;32m{s}\033[0m"
logt = lambda: "\033[01;32m{%d}\033[0m" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

conf_thres, iou_thres = 0.25, 0.6


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='stylegan2')
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--mix_prob", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--milestones", type=int, nargs='+', default=[50, 100])

    parser.add_argument('--obj_model', type=str, default="data/models/vehicle-YZ.obj")
    parser.add_argument('--selected_faces', type=str, default="data/models/selected_faces.txt")
    parser.add_argument('--texture_size', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=1024)

    parser.add_argument('--categories', type=str, nargs='+', default=["dog"])
    parser.add_argument('--pretrained', type=str)

    args: ArgsType = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


class ArgsType:
    device: str
    save_dir: str
    epochs: int
    batch: int
    num_workers: int
    size: int

    mix_prob: float
    lr: float
    milestones: List[int]

    obj_model: str
    selected_faces: str
    texture_size: int
    latent_dim: int

    categories: List[str]
    pretrained: str


def prepare_training(args: ArgsType):
    # ----------------------------------------------
    #   Load Data
    # ----------------------------------------------
    train_set = CarlaDataset(carla_root="tmp/data", categories=args.categories)
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch,
        num_workers=args.num_workers,
        collate_fn=CarlaDataset.collate_fn,
        drop_last=True
    )
    num_classes = len(train_set.coco_ic_map)
    # --- Load Neural Renderer ---
    with open(args.selected_faces, 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model,
        selected_faces=selected_faces,
        texture_size=args.texture_size,
        image_size=800,
        device=args.device,
    )

    # --- Load Texture Generator ---
    model = TextureGenerator(
        nt=len(selected_faces),
        ts=args.texture_size,
        style_dim=args.latent_dim,
        cond_dim=num_classes,
        mix_prob=args.mix_prob,
    )
    model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.cuda().train()
    model.encoder.eval()

    optimized_params = [{"params": model.generator.parameters()}, {"params": model.decoder.parameters()}]
    # optimizer = optim.Adam(optimized_params, lr=args.lr,weight_decay=1e-4)
    optimizer = optim.SGD(optimized_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    lr_heduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    # --- Load Detector ---
    with open("configs/hyp.scratch-low.yaml", "r") as f:
        hyp: dict = yaml.safe_load(f)
    detector = Model("configs/yolov5s.yaml", ch=3, nc=num_classes, anchors=hyp.get('anchors')).to(args.device)
    detector.nc = num_classes
    detector.hyp = hyp
    detector_loss = ComputeLoss(detector)
    ckpt = torch.load("pretrained/yolov5s.pt", map_location='cpu')
    detector.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
    detector.eval()

    return (neural_renderer, model, detector, detector_loss, optimizer, lr_heduler, train_loader)


def train():
    args = get_args()
    print(cstr(args))
    nowt = datetime.datetime.now().strftime("%m%d%H%M")
    args.save_dir = args.save_dir + "-" + nowt

    os.makedirs(sample_save_dir := os.path.join("tmp", args.save_dir, "sample"), exist_ok=True)
    os.makedirs(checkpoint_save_dir := os.path.join("tmp", args.save_dir, "checkpoint"), exist_ok=True)
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s] %(message)s',
        level=logging.INFO,
        filename=os.path.join("tmp", args.save_dir, f'train.log'),
        filemode='a',
    )

    (neural_renderer, model, detector, detector_loss, optimizer, lr_heduler, train_loader) = prepare_training(args)
    device = args.device

    n_r = 0.001

    # ----------------------------------------------
    #   start training
    # ----------------------------------------------
    for epoch in range(args.epochs):
        clr = optimizer.state_dict()['param_groups'][0]['lr']
        print(
            "\033[32m", f"[Epoch]{epoch}/{args.epochs}", f"[Batch Size]{args.batch}", f"[LR]{clr:.6f}", f":{device}",
            "\033[0m"
        )

        loss_smooth_epoch = 0
        ldet_epoch, lbox_epoch, lobj_epoch, lcls_epoch = 0, 0, 0, 0
        ladv_epoch = 0
        data_num = 0
        loss_zero = torch.tensor(0).to(device)

        accum = 0.5**(32 / (10 * 1000))

        pbar = tqdm.tqdm(train_loader)
        model.generator.train()
        if True:
            for i_mb, items in enumerate(pbar):
                images = items["image"].to(device)
                labels = items["label"].to(device)
                bs = images.shape[0]
                data_num += bs

                tt = neural_renderer.textures[:, neural_renderer.selected_faces, :]
                xs_t = tt.repeat(args.batch, *[1] * (len(tt.size()) - 1))
                xs_n = torch.rand_like(xs_t)         # x_{noise}
                xs_i = (1 - n_r) * xs_t + n_r * xs_n # x_{texture with noise}
                xs_adv = model.decode(model.forward(xs_i, labels))

                adv_textures = neural_renderer.textures.repeat(
                    xs_adv.shape[0], *[1] * (len(neural_renderer.textures.size()) - 1)
                )
                adv_textures[:, neural_renderer.selected_faces, :] = xs_adv

                render_scene_list, render_label_list = [], []
                for i_b in range(bs):
                    r_p = {"ct": items["ct"][i_b], "vt": items["vt"][i_b], "fov": items["fov"][i_b]}
                    adv_texture = adv_textures[i_b].unsqueeze(0)
                    neural_renderer.set_render_perspective(r_p["ct"], r_p["vt"], r_p["fov"])
                    rgb_image, _, alpha_image = neural_renderer.forward(F.tanh(adv_texture))
                    render_image = alpha_image * rgb_image + (1 - alpha_image) * images[i_b]
                    render_scene_list.append(render_image)

                    # find object label
                    binary = np.ascontiguousarray(alpha_image.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    find_boxes = []
                    for c in contours:
                        [x, y, w, h] = cv2.boundingRect(c)
                        find_boxes.append([x, y, x + w, y + h])
                    fc = np.array(find_boxes)

                    box = [min(fc[:, 0]), min(fc[:, 1]), max(fc[:, 2]), max(fc[:, 3])] # [x1,y1,x2,y2]

                    if False:
                        scene_img: np.ndarray = np.ascontiguousarray(render_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)   # yapf: disable
                        [x1, y1, x2, y2] = box
                        cv2.rectangle(scene_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = train_set.coco_ic_map[int(labels[i_b])]
                        cv2.putText(scene_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imwrite('tmp/render.png', scene_img)
                        exit()

                    b, c, h, w = rgb_image.shape
                    render_label_list.append(torch.tensor([
                        i_b, int(labels[i_b]),
                        (box[0] + box[2]) / 2 / w,
                        (box[1] + box[3]) / 2 / h,
                        (box[2] - box[0]) / w,
                        (box[3] - box[1]) / h,
                    ])) # yapf:disable
                render_scenes = torch.cat(render_scene_list).to(device)
                render_labels = torch.stack(render_label_list).to(device)

                _, pred = detector.forward(render_scenes)
                (ldet, (lbox, lobj, lcls)) = detector_loss.__call__(pred, render_labels)
                lbox *= 0.05
                lobj *= 1.0
                lcls *= 0.5 * 1.5
                lbox_epoch += lbox.item() * bs
                lobj_epoch += lobj.item() * bs
                lcls_epoch += lcls.item() * bs
                ldet_epoch += ldet.item() * bs

                if i_mb % 4 == 0:
                    loss_smooth = 0.05 * F.l1_loss(xs_t, xs_adv)
                    loss_smooth_epoch += loss_smooth.item() * bs / 16
                else:
                    loss_smooth = loss_zero

                loss_adv = lbox + lobj + lcls # + loss_smooth

                ladv_epoch += (lbox + lobj + lcls).item() * bs + loss_smooth * bs * 16

                optimizer.zero_grad()
                loss_adv.backward()
                # clip_grad_norm_(parameters=g_model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()

                # accumulate(g_ema, generator, accum)

                pbar.set_description(" ".join((
                    f"{cstr('adv')}:{loss_adv.item():.4f}",
                    f"{cstr('ls')}:{loss_smooth.item():.4f}",
                    f"{cstr('ldet')}:{ldet.item():.4f}",
                    f"{cstr('lcls')}:{lcls.item():.4f}",
                    f"{cstr('lbox')}:{lbox.item():.4f}",
                    f"{cstr('lobj')}:{lobj.item():.4f}",
                )))# yapf:disable
        lr_heduler.step()

        #  Valid
        if epoch % 10 == 0:
            model.generator.eval()

            x_t = neural_renderer.textures[:, neural_renderer.selected_faces, :]
            train_set = train_loader.dataset
            random_indexs = random.sample(range(len(train_set)), 6)
            pbar = tqdm.tqdm(random_indexs)
            result_imgs = []
            for i_d, index in enumerate(pbar):
                item = train_set[index]
                image = item["image"].to(device)
                label = item["label"].to(device)
                r_p = {"ct": item["ct"], "vt": item["vt"], "fov": item["fov"]}

                x_n = torch.rand_like(x_t)                          # x_{noise}
                x_i = (1 - n_r) * x_t + n_r * x_n                   # x_{texture with noise}
                with torch.no_grad():
                    x_adv = model.decode(model.forward(x_i, label)) # x_{adv}

                def get_pred_img(tt: torch.Tensor):
                    render_image, _, _, render_img = render_a_image(neural_renderer, tt, image, r_p)
                    with torch.no_grad():
                        eval_pred, train_preds = detector.forward(render_image) # real
                        pred_results = non_max_suppression(eval_pred, conf_thres, iou_thres, None, False)[0]
                    if len(pred_results):
                        for *xyxy, conf, cls in pred_results:
                            label = '%s %.2f' % (train_set.coco_ic_map[int(cls)], conf)
                            x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                            cv2.rectangle(render_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(render_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    return render_img

                result_imgs.append(
                    cv2.vconcat([
                        get_pred_img(x_t),
                        get_pred_img(x_i),
                        get_pred_img(x_adv),
                        get_pred_img(x_n),
                    ])
                )
                pbar.set_description(f"{i_d}: sample carla[{index}]")

            concat_image = cv2.hconcat(result_imgs)
            cv2.imwrite(os.path.join(sample_save_dir, f'_detect.png'), concat_image)
            cv2.imwrite(os.path.join(sample_save_dir, f'detect-{epoch}.png'), concat_image)
            # [
            #     cv2.imwrite(os.path.join(sample_save_dir, f'detect-{epoch}_{i_img}.png'), result_imgs[i])
            #     for i_img in range(len(result_imgs))
            # ]

        epoch_loss_dict = {
            "LAdv": ladv_epoch / data_num,
            "LDet": ldet_epoch / data_num,
            "Ls": loss_smooth_epoch / data_num / 16,
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
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
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


def render_a_image(
    neural_renderer: NeuralRenderer, x_texture: torch.Tensor, base_image: torch.Tensor, render_params: dict
):
    tt_adv = neural_renderer.textures
    tt_adv[:, neural_renderer.selected_faces, :] = x_texture
    neural_renderer.set_render_perspective(render_params["ct"], render_params["vt"], render_params["fov"])
    rgb_image, _, alpha_image = neural_renderer.forward(F.tanh(tt_adv))
    render_image = alpha_image * rgb_image + (1 - alpha_image) * base_image
    render_img = np.ascontiguousarray(render_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255)
    return render_image, rgb_image, alpha_image, render_img.astype(np.uint8)


if __name__ == "__main__":
    train()
