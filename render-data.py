"""
渲染车辆到全部数据集，筛选出渲染歪的
"""
import abc
from typing import Dict, List
import os, sys
import json
import argparse
import cv2
import numpy as np

import torch
import tqdm
import tsgan
from tsgan.render import NeuralRenderer
from tsgan import types
import neural_renderer as nr


class TypeArgs(metaclass=abc.ABCMeta):
    obj_model: str
    texture_size: int
    data_dir: str
    device: str


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model', type=str, default='data/models/vehicle-YZ.obj')
    parser.add_argument('--texture_size', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default="tmp/data")
    parser.add_argument('--device', type=str, default='cuda')
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
    args: TypeArgs = get_args()

    # Load Neural Renderer
    neural_renderer = NeuralRenderer(args.obj_model, texture_size=4, image_size=800, device=args.device)
    neural_renderer.to(neural_renderer.textures.device)

    # Load Image and label
    label_dir = os.path.join(args.data_dir, "labels")
    image_dir = os.path.join(args.data_dir, "images")
    render_dir=os.path.join(args.data_dir, "render")
    os.makedirs(render_dir, exist_ok=True)
    pbar=tqdm.tqdm(os.listdir(label_dir))
    for file in pbar:
        try:
            with open(os.path.join(label_dir, file), 'r') as f:
                label_dict = json.load(f)
        except:
            print(f"not found label {os.path.join(label_dir,file)}")
            continue

        vehicle_transform = convert_dict_transform(label_dict['vehicle'])
        camera_transform = convert_dict_transform(label_dict['camera'])
        fov = label_dict['camera']['fov']
        name = label_dict['name']

        # Load Image
        image_file=os.path.join(image_dir, f"{name}.png")
        if not os.path.exists(image_file):
            print(f"not found image {image_file}")
            continue

        render_file=os.path.join(render_dir, f"{name}.png")
        if os.path.exists(render_file):
            print(f"rendered {image_file}")
            continue
        image = cv2.imread(image_file)

        # render
        neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)
        (
            rgb_images,
            depth_images,
            alpha_images,
        ) = neural_renderer.renderer.forward(
            neural_renderer.vertices,
            neural_renderer.faces,
            torch.tanh(neural_renderer.textures),
        )
        rgb_image: torch.Tensor = rgb_images[0]
        rgb_img: np.ndarray = rgb_image.detach().cpu().numpy() * 255
        rgb_img = rgb_img.transpose(1, 2, 0)

        alpha_image: torch.Tensor = alpha_images[0]
        alpha_channel: np.ndarray = alpha_image.detach().cpu().numpy()

        render_image = np.zeros(rgb_img.shape)
        for x in range(alpha_channel.shape[0]):
            for y in range(alpha_channel.shape[1]):
                alpha = alpha_channel[x][y]
                render_image[x][y] = alpha * rgb_img[x][y] + (1 - alpha) * image[x][y]
        cv2.imwrite(render_file, render_image)


if __name__ == '__main__':
    main()
