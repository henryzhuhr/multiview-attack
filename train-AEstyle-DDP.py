import math
from typing import Dict, List
import os, sys
import json
import argparse

import tqdm
import cv2
import numpy as np

import torch
from torch import Tensor, optim, nn
from torch import distributed
import tsgan
from tsgan.render import NeuralRenderer
from tsgan import types
import neural_renderer as nr


# TODO: 多卡训练 https://www.bilibili.com/video/BV1xZ4y1S7dG
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model', type=str, default='data/models/vehicle-YZ.obj')
    parser.add_argument('--autoencoder_pretrained', type=str, default='tmp/autoencoder.pt')
    parser.add_argument('--texture_size', type=int, default=4)
    scence_name = 'Town10HD-point_0000-distance_000-direction_1'
    parser.add_argument('--scence_image', type=str, default=f'tmp/data/images/{scence_name}.png')
    parser.add_argument('--scence_label', type=str, default=f'tmp/data/labels/{scence_name}.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='tmp/autoencoder')
    parser.add_argument('--save_name', type=str, default='autoencoder')

    parser.add_argument('--epoches', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--milestones', type=List[int], default=[3000, 10000])

    # for distributed, python -m torch.distributed.launch set it automatically
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Distributed ----
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1
    if args.local_rank == 0:
        print(f"Run in {n_gpu} GPUs")
    if args.distributed:
        torch.cuda.set_device(args.local_rank) # set gpu used in current processing
        distributed.init_process_group(        # init distributed environment
            backend='nccl',
            init_method='env://',
            world_size=n_gpu,
            rank=args.local_rank,
        )

    # ---- Load Pretrained AutoEncoder ----
    num_points = 12306
    num_feature = 4
    latent_dim = 256
    encoder = tsgan.models.autoencoder.TextureEncoder(
        num_feature=num_feature,
        latent_dim=latent_dim,
    )
    decoder = tsgan.models.autoencoder.TextureDecoder(
        latent_dim=latent_dim,
        num_points=num_points,
        num_feature=num_feature,
    )
    autoencoder_pretrained = torch.load(args.autoencoder_pretrained, map_location="cpu")
    encoder.load_state_dict(autoencoder_pretrained['encoder'])
    decoder.load_state_dict(autoencoder_pretrained['decoder'])
    encoder.eval()
    decoder.eval()
    

    if args.distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder.cuda(args.local_rank), device_ids=[args.local_rank])
        decoder = torch.nn.parallel.DistributedDataParallel(decoder.cuda(args.local_rank), device_ids=[args.local_rank])
    else:
        encoder = encoder.to(args.device)
        decoder = decoder.to(args.device)


if __name__ == '__main__':
    main()
