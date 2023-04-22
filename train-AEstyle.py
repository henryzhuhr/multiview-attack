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

from tsgan.models import stylegan2

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

    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Load Texture and Pretrained AutoEncoder ----
    num_points = 12306
    num_feature = 4
    args.latent_dim=256
    with open('data/models/selected_faces.txt', 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model,
        selected_faces=selected_faces,
        texture_size=num_feature,
        image_size=800,
        device=args.device,
    )
    neural_renderer.to(neural_renderer.textures.device)
    neural_renderer.renderer.to(neural_renderer.textures.device)
    # neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)
    print('get textures size:', neural_renderer.textures.size())

    encoder = tsgan.models.autoencoder.TextureEncoder(
        num_feature=num_feature,
        latent_dim=args.latent_dim,
    )
    decoder = tsgan.models.autoencoder.TextureDecoder(
        latent_dim=args.latent_dim,
        num_points=num_points,
        num_feature=num_feature,
    )
    autoencoder_pretrained = torch.load(args.autoencoder_pretrained, map_location="cpu")
    encoder.load_state_dict(autoencoder_pretrained['encoder'])
    decoder.load_state_dict(autoencoder_pretrained['decoder'])
    encoder.eval()
    decoder.eval()
    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)

    texture_latent=encoder.forward(neural_renderer.textures)
    print('texture_latent size:', texture_latent.size())

    args.size=1024
    args.n_mlp=8
    generator = stylegan2.Generator(
        args.size,
        args.latent_dim,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier,
    ).to(args.device)
    
if __name__ == '__main__':
    main()
