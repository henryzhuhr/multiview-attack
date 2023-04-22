import os
import argparse
import sys
import tqdm

import torch
from torch import Tensor, nn
from torch.utils import data
from torch.optim.lr_scheduler import LambdaLR


import tsgan

# import torch.utils.tensorboard

from tsgan.render import NeuralRenderer

from data import CarlaDataset
# from modules.models.diffusion import TextureDiffusion


def get_args():
    parser = argparse.ArgumentParser()
    # dataset and model
    parser.add_argument('--dataroot', type=str, default='/home/zhr/project/carla-project/tmp/data')
    parser.add_argument('--obj_model', type=str, default='data/models/vehicle-YZ.obj')
    parser.add_argument('--selected_faces', type=str, default='data/models/selected_faces.txt')
    
    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--sched_start_epoch', type=int, default=150 * 10e3)
    parser.add_argument('--sched_end_epoch', type=int, default=300 * 10e3)
    parser.add_argument('--end_lr', type=float, default=1e-4)

    # Training
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--max_iters', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    return parser.parse_args()


def main():
    args = get_args()
    device = args.device

    texture_size = 4

    # -- load dataset
    croppedcoco_train_set=tsgan.data.CroppedCOCO(config_file='configs/coco.yaml',is_train=True)
    croppedcoco_train_loader = data.DataLoader(croppedcoco_train_set, batch_size=args.batch_size, num_workers=args.num_workers)
    croppedcoco_valid_set=tsgan.data.CroppedCOCO(config_file='configs/coco.yaml',is_train=False)    
    croppedcoco_valid_loader = data.DataLoader(croppedcoco_valid_set, batch_size=args.batch_size, num_workers=args.num_workers)

    # -- load texture and renderer
    texture_size = 4
    with open(args.selected_faces, 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model,
        selected_faces=selected_faces,
        texture_size=texture_size,
        image_size=800,
        device=device,
    )
    neural_renderer.to(neural_renderer.textures.device)
    neural_renderer.renderer.to(neural_renderer.textures.device)
    texture_inputs = neural_renderer.textures

    batch_textures = texture_inputs.repeat(args.batch_size, 1, 1, 1, 1, 1)
    print("texture size:        ", texture_inputs.size())
    print("batch textures size: ", batch_textures.size())

   

if __name__ == "__main__":
    main()
