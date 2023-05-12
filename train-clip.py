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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]                         # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))                 # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative

from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import non_max_suppression

cstr = lambda s: f"\033[01;32m{s}\033[0m"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='attack')
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)

    parser.add_argument('--obj_model', type=str, default="data/models/vehicle-YZ.obj")
    parser.add_argument('--selected_faces', type=str, default="data/models/faces-std.txt")
    parser.add_argument('--texture_size', type=int, default=4)
    return parser.parse_args()


def prepare(args):
    mix_train_set = tsgan.data.CroppedCOCOCarlaMixDataset(
        'configs/dataset.yaml',
        is_train=False,                                    # TODO: 测试完后, False 修改为训练 True
        show_detail=True,
        transform=transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])
    )

    # --- Load Neural Renderer
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
    npoint = x_t.shape[1]
    print('textures num:%d (%d selected)' % (neural_renderer.textures.shape[1], npoint))
    # --- Detector
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


    return (neural_renderer, detector,  detector_loss, mix_train_set)


def main():
    args = get_args()
    nowt = datetime.datetime.now().strftime("%m%d%H%M")
    args.save_dir = f"{args.save_dir}-{nowt}"
    (neural_renderer, detector, detector_loss, mix_train_set) = prepare(args)

    os.makedirs(sample_save_dir := os.path.join("tmp", args.save_dir, "sample"), exist_ok=True)
    os.makedirs(checkpoint_save_dir := os.path.join("tmp", args.save_dir, "checkpoint"), exist_ok=True)
    logfilename = os.path.join("tmp", args.save_dir, f'train.log')
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO, filename=logfilename, filemode='a'
    )

    t_mask = neural_renderer.textures_mask.unsqueeze(0)
    optimizer = torch.optim.Adam(neural_renderer.parameters(), lr=args.lr, betas=(0.5,0.999))


    for epoch in range(args.epochs):
        print("\033[32m", f"[Epoch]{epoch}/{args.epochs}", "\033[0m") # yapf: disable
        pbar = tqdm(mix_train_set)

        ladv_epoch = 0
        ldet_epoch = 0
        lcls_epoch = 0
        lbox_epoch = 0
        lobj_epoch = 0
        for data_dict in pbar:
            coco_label = data_dict["coco"]["predict_id"]
            crp = data_dict["carla"] # carla_render_params

            textures = (1 -
                        t_mask) * neural_renderer.textures + t_mask * (F.tanh(neural_renderer.render_textures) + 1) / 2

            neural_renderer.set_render_perspective(crp["camera_transform"], crp["vehicle_transform"], crp["fov"])
            (rgb_image, _, alpha_image) = neural_renderer.forward(textures)

            alpha_img = (alpha_image.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)

            contours, _ = cv2.findContours(alpha_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            find_boxes = []
            for c in contours:
                [x, y, w, h] = cv2.boundingRect(c)
                find_boxes.append([x, y, x + w, y + h])
            fc = np.array(find_boxes)
            box = [min(fc[:, 0]), min(fc[:, 1]), max(fc[:, 2]), max(fc[:, 3])]                                  # [x1,y1,x2,y2]
            label = torch.tensor(
                [
                    0, coco_label, (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h, (box[2] - box[0]) / w,
                    (box[3] - box[1]) / h
                ]
            ).unsqueeze(0).cuda()
            _, pred = detector.forward(rgb_image)
            (ldet, (lbox, lobj, lcls)) = detector_loss.__call__(pred, label)
            lcls *= 1.5
            lbox *= 0.05
            lobj *= 1.0
            adv_loss = lcls + lbox + lobj

            ladv_epoch += adv_loss.item()
            ldet_epoch += ldet.item()
            lcls_epoch += lcls.item()
            lbox_epoch += lbox.item()
            lobj_epoch += lobj.item()

            optimizer.zero_grad()
            adv_loss.backward()
            optimizer.step()

            pbar.set_description(" ".join((
                f"{cstr('adv_loss')}:{adv_loss.item():.4f}",
                f"{cstr('det')}:{ldet.item():.4f}",
                f"{cstr('lcls')}:{lcls.item():.4f}",
                f"{cstr('lbox')}:{lbox.item():.4f}",
                f"{cstr('lobj')}:{lobj.item():.4f}",
            ))) # yapf: disable

        epoch_loss_dict = {
            "Adv": ladv_epoch / len(mix_train_set),
            "Det": ldet_epoch / len(mix_train_set),
            "lcls": lcls_epoch / len(mix_train_set),
            "lbox": lbox_epoch / len(mix_train_set),
            "lobj": lobj_epoch / len(mix_train_set),
        }

        logging.info(" ".join(["epoch:%-4d" % epoch] + [f"{k}:{v:.5f}" for k, v in epoch_loss_dict.items()]))
        print(" ".join([f"\033[00;34m{k}\033[0m:{v:.5f}" for k, v in epoch_loss_dict.items()]))
        print()

        result_imgs = []
        for i_d, data_dict in enumerate(mix_train_set):
            if i_d > 10:
                break
            scene_img = data_dict["carla"]["image"]
            crp = data_dict["carla"]                                                                    # carla render params
            scene_image = torch.from_numpy(scene_img).to(neural_renderer.textures.device).permute(2, 0,
                                                                                                  1).float() / 255.

            textures = (1 -
                        t_mask) * neural_renderer.textures + t_mask * (F.tanh(neural_renderer.render_textures) + 1) / 2
            neural_renderer.set_render_perspective(crp["camera_transform"], crp["vehicle_transform"], crp["fov"])

            def get_img(tt, t_s_image):
                (rgb_image, _, alpha_image) = neural_renderer.forward(tt)

                t_render_image = alpha_image * rgb_image + (1 - alpha_image) * t_s_image
                with torch.no_grad():
                    pred_infer, pred_train = detector.forward(t_render_image)

                conf_thres, iou_thres = 0.25, 0.6

                det = non_max_suppression(pred_infer, conf_thres, iou_thres, None, False)[0]
                cv_img = np.ascontiguousarray(t_render_image[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)#yapf:disable
                if len(det):
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (mix_train_set.COCO_CLASS[int(cls)], conf)
                        x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(cv_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return cv_img

            clean_img = get_img(neural_renderer.textures, scene_image)
            attack_img = get_img(textures, scene_image)
            result_imgs.append(cv2.hconcat([clean_img, attack_img]))
        result_img = cv2.vconcat(result_imgs)

        cv2.imwrite(os.path.join(sample_save_dir, f'detect-{epoch}.png'), result_img)
        cv2.imwrite(os.path.join(sample_save_dir, f'__detect.png'), result_img)

        torch.save(
            {
                "render_textures": neural_renderer.render_textures,
                "args": args
            }, f"{checkpoint_save_dir}/gan-{epoch}.pt"
        )


if __name__ == "__main__":
    main()