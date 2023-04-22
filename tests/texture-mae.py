from typing import Dict
import os, sys
import json
import argparse

import tqdm
import cv2
import numpy as np

import torch
from torch import Tensor, optim

sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
from tsgan.data.dataset import CarlaDataset
from tsgan.render import NeuralRenderer
from tsgan import types

import neural_renderer as nr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model', type=str, default='data/models/vehicle-YZ.obj')
    scence_name = 'Town10HD-point_0000-distance_001-direction_1'
    parser.add_argument('--scence_image', type=str, default=f'data/dataset/images/{scence_name}.png')
    parser.add_argument('--scence_label', type=str, default=f'data/dataset/labels/{scence_name}.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='tmp')
    return parser.parse_args()


def render_image():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载渲染器
    Ts = 4
    with open('data/models/selected_faces.txt', 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model, selected_faces=selected_faces, texture_size=Ts, image_size=800, device=args.device
    )
    neural_renderer.to(neural_renderer.textures.device)
    neural_renderer.renderer.to(neural_renderer.textures.device)
    print('textures: ', neural_renderer.textures.size())

    image = cv2.imread(args.scence_image)
    with open(args.scence_label, 'r') as f:
        label_dict = json.load(f)

        def convert_dict_transform(transform_dict: Dict):
            return types.carla.Transform(
                location=types.carla.Location(
                    x=transform_dict['location']['x'],
                    y=transform_dict['location']['y'],
                    z=transform_dict['location']['z']
                ),
                rotation=types.carla.Rotation(
                    pitch=transform_dict['rotation']['pitch'],
                    yaw=transform_dict['rotation']['yaw'],
                    roll=transform_dict['rotation']['roll']
                )
            )

    vehicle_transform = convert_dict_transform(label_dict['vehicle'])
    camera_transform = convert_dict_transform(label_dict['camera'])
    fov = label_dict['camera']['fov']
    name = label_dict['name']

    neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)

    # ==========================
    model = MAE(
        texture_size=Ts,
        num_points=neural_renderer.textures.size(1),
        mask_ratio=0.75,
    )
    model.to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    model.forward(torch.stack([
        neural_renderer.textures.squeeze(0),
        neural_renderer.textures.squeeze(0),
    ]).to(args.device))
    return
    for i in range(1000):

        x = neural_renderer.textures
        optimizer.zero_grad()
        (
            new_texture,
            loss,
        ) = model.forward(x)

        expoch_loss = loss.item()
        loss.backward()
        optimizer.step()

        print("iter:%d loss:%.4f" % (i, expoch_loss))

        (
            rgb_images,
            depth_images,
            alpha_images,
        ) = neural_renderer.renderer.forward(
            neural_renderer.vertices,
            neural_renderer.faces,
            torch.tanh(new_texture),
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

        name = "render-car"
        cv2.imwrite(os.path.join(args.save_dir, f'{name}.png'), render_image)

    # ==========================


if __name__ == '__main__':
    render_image()
