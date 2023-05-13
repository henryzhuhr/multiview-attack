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
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--mix_prob", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--lr", type=float, default=0.002)

    parser.add_argument('--obj_model', type=str, default="data/models/vehicle-YZ.obj")
    parser.add_argument('--selected_faces', type=str, default="data/models/faces-std.txt")
    parser.add_argument('--texture_size', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=1024)
    # parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default="tmp/generator-bowl/checkpoint/gan-268.pt")

    return parser.parse_args()


def main():
    args: ArgsType = get_args()
    data_set = tsgan.data.CarlaDataset(carla_root="tmp/data", categories=["dog", "car"])

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
    # --- Load Texture Generator ---
    model = TextureGenerator(
        nt=len(selected_faces),
        ts=args.texture_size,
        style_dim=args.latent_dim,
        cond_dim=len(data_set.coco_ic_map),
        mix_prob=args.mix_prob
    )
    pretrained = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(pretrained["g"])
    model.cuda().eval()

    # --- Load Detector ---
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

    x_t = neural_renderer.textures[:, neural_renderer.selected_faces, :]
    print(x_t.shape)

    for item in data_set:
        image = item["image"]
        label = item["label"]
        # (vt,ct,fov)=item


if __name__ == "__main__":
    main()