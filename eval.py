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


if __name__ == "__main__":
    main()