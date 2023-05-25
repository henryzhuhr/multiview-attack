import abc
from typing import Dict, List
import os, sys
import json
import argparse

import tqdm
import cv2
import numpy as np

import torch
from torch import Tensor, optim, nn
from torch.nn import functional as F

from models.data.carladataset import CarlaDataset
from models.render import NeuralRenderer
from models.gan import TextureGenerator



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model', type=str, default='assets/audi_et_te.obj')
    parser.add_argument('--selected_faces', type=str, default='assets/faces-std.txt')
    parser.add_argument('--texture_size', type=int, default=4)
    parser.add_argument('--scence_image', type=str, default="images/carla-scene.png")
    parser.add_argument('--scence_label', type=str, default="images/carla-scene.json")

    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

class TypeArgs(metaclass=abc.ABCMeta):
    __metaclass__ = abc.ABCMeta
    obj_model: str
    selected_faces: str
    texture_size: int
    scence_image: str
    scence_label: str
def main():
    args: TypeArgs = get_args()
    # Load Image and label
    image = cv2.imread(args.scence_image)
    with open(args.scence_label, 'r') as f:
        label_dict = json.load(f)
        vehicle_transform = CarlaDataset.convert_dict_transform(label_dict['vehicle'])
        camera_transform = CarlaDataset.convert_dict_transform(label_dict['camera'])
        fov = label_dict['camera']['fov']
        name = label_dict['name']

    # Load Neural Renderer
    ts = args.texture_size
    with open(args.selected_faces, "r") as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model, selected_faces=selected_faces, texture_size=ts, image_size=800, device=args.device
    )
    neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)

    x_t = neural_renderer.textures[:, selected_faces, :]
    img=render_a_image(neural_renderer, image, x_t)
    cv2.imwrite("images/test.png", img)


def render_a_image(neural_renderer: NeuralRenderer, image: cv2.Mat, x: Tensor):
    # x_full = torch.zeros_like(neural_renderer.textures)
    x_full = neural_renderer.textures
    x_full[:, neural_renderer.selected_faces, :] = x

    rgb_images, _, alpha_images = neural_renderer.renderer.forward(
        neural_renderer.vertices, neural_renderer.faces, torch.tanh(x_full)
    )
    rgb_img: cv2.Mat = (rgb_images[0].detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    alpha_channel: cv2.Mat = alpha_images[0].detach().cpu().numpy()

    render_image = np.zeros(rgb_img.shape)
    for x in range(alpha_channel.shape[0]):
        for y in range(alpha_channel.shape[1]):
            alpha = alpha_channel[x][y]
            render_image[x][y] = alpha * rgb_img[x][y] + (1 - alpha) * image[x][y]
    return render_image


if __name__ == '__main__':
    main()
