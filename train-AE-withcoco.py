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

import tsgan
from tsgan.render import NeuralRenderer
from tsgan import types
import neural_renderer as nr
from tsgan.models.autoencoder import (TextureEncoder, TextureDecoder)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model', type=str, default='data/models/vehicle-YZ.obj')
    parser.add_argument('--texture_size', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=2048)
    parser.add_argument('--scence_image', type=str, default="images/carla-scene.png")
    parser.add_argument('--scence_label', type=str, default="images/carla-scene.json")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='tmp/autoencoder')
    parser.add_argument('--save_name', type=str, default='autoencoder')

    parser.add_argument('--epoches', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--milestones', type=List[int], default=[3000, 10000])
    return parser.parse_args()


def convert_dict_transform(transform_dict: Dict):
    return types.carla.Transform(
        location=types.carla.Location(
            x=transform_dict['location']['x'], y=transform_dict['location']['y'], z=transform_dict['location']['z']
        ),
        rotation=types.carla.Rotation(
            pitch=transform_dict['rotation']['pitch'],
            yaw=transform_dict['rotation']['yaw'],
            roll=transform_dict['rotation']['roll']
        )
    )


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Load Image and label
    image = cv2.imread(args.scence_image)
    with open(args.scence_label, 'r') as f:
        label_dict = json.load(f)
        vehicle_transform = convert_dict_transform(label_dict['vehicle'])
        camera_transform = convert_dict_transform(label_dict['camera'])
        fov = label_dict['camera']['fov']
        name = label_dict['name']

    # Load Neural Renderer
    Ts: int = args.texture_size
    with open('data/models/selected_faces.txt', 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
        selected_faces = None
    neural_renderer = NeuralRenderer(
        args.obj_model, selected_faces=selected_faces, texture_size=Ts, image_size=800, device=args.device
    )
    neural_renderer.to(neural_renderer.textures.device)
    neural_renderer.renderer.to(neural_renderer.textures.device)
    neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)
    print('textures: ', neural_renderer.textures.size())

    # Load Model
    ts = args.texture_size
    num_points = neural_renderer.textures.shape[1]
    latent_dim = args.latent_dim
    encoder = TextureEncoder(num_feature=ts, latent_dim=latent_dim).to(args.device)
    decoder = TextureDecoder(latent_dim=latent_dim, num_points=num_points, num_feature=ts).to(args.device)

    criterion = nn.L1Loss().to(args.device)

    optimizer = optim.SGD(
        [{
            'params': encoder.parameters()
        }, {
            'params': decoder.parameters()
        }],
        lr=1e-1,
    )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1)

    encoder.train()
    decoder.train()
    pabr = tqdm.tqdm(range(args.epoches))
    for i in pabr:
        x = neural_renderer.textures
        optimizer.zero_grad()
        # --- forward ---
        latent_x = encoder.forward(x)
        rec_x = decoder.forward(latent_x)
        loss = criterion.forward(x, rec_x)

        expoch_loss = loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        pabr.set_description("iter:%d loss:%.8f latent: %s" % (i, expoch_loss, str(latent_x.size())))

        # if ((i < 5000) and (i % 500 == 0)) or (i % 10000 == 0):
        if i % 500 == 0:
            print("iter:%d loss:%.4f" % (i, expoch_loss))
            (
                rgb_images,
                depth_images,
                alpha_images,
            ) = neural_renderer.renderer.forward(
                neural_renderer.vertices,
                neural_renderer.faces,
                torch.tanh(rec_x),
            )

            rgb_image: Tensor = rgb_images[0]
            rgb_img: np.ndarray = rgb_image.detach().cpu().numpy() * 255
            rgb_img = rgb_img.transpose(1, 2, 0)

            alpha_image: Tensor = alpha_images[0]
            alpha_channel: np.ndarray = alpha_image.detach().cpu().numpy()

            render_image = np.zeros(rgb_img.shape)
            for x in range(alpha_channel.shape[0]):
                for y in range(alpha_channel.shape[1]):
                    alpha = alpha_channel[x][y]
                    render_image[x][y] = alpha * rgb_img[x][y] + (1 - alpha) * image[x][y]

            save_name = "autoencoder"
            cv2.imwrite(os.path.join(args.save_dir, f'{save_name}.png'), render_image)
            torch.save(
                {'encoder':encoder.state_dict(),'decoder':decoder.state_dict(),},
                os.path.join(args.save_dir, f'{args.save_name}.pt'),
            )
            cv2.imwrite(os.path.join(args.save_dir, f'{save_name}-{i}.png'), render_image)
            torch.save(
                {'encoder':encoder.state_dict(),'decoder':decoder.state_dict(),},
                os.path.join(args.save_dir, f'{args.save_name}-{i}.pt'),
            )


if __name__ == '__main__':
    main()