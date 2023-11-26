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

import neural_renderer as nr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='tmp/train/physicalAttak-dog-06211610/checkpoint/_generator.pt')
    parser.add_argument('--scence_name', type=str, default="data/samples/image~0_0_4_90")

    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


class TypeArgs(metaclass=abc.ABCMeta):
    __metaclass__ = abc.ABCMeta
    obj_model: str
    selected_faces: str
    texture_size: int
    scence_name: str


def main():
    args: TypeArgs = get_args()
    device = torch.device(args.device)

    pretrained = torch.load(args.pretrained, map_location='cpu')
    pargs = vars(pretrained["args"]) # pretrained args
    ts = pargs["texture_size"]
    cats: List[str] = pargs["categories"]
    obj_model = pargs["obj_model"]
    latent_dim = pargs["latent_dim"]
    mix_prob = pargs["mix_prob"]

    image = cv2.imread(args.scence_name+".png")
    with open(args.scence_name+".json", 'r') as f:
        label_dict = json.load(f)
        vehicle_transform = CarlaDataset.convert_dict_transform(label_dict['vehicle'])
        camera_transform = CarlaDataset.convert_dict_transform(label_dict['camera'])
        fov = label_dict['camera']['fov']
        name = label_dict['name']

    # Load Neural Renderer
    with open(pargs["selected_faces"], "r") as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        obj_model, selected_faces=selected_faces, texture_size=ts, image_size=800,batch_size=8, device=device
    )
    neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)

    x_t = neural_renderer.textures[:, selected_faces, :]

    model = TextureGenerator(
        nt=len(neural_renderer.selected_faces), ts=ts, style_dim=latent_dim, cond_dim=80, mix_prob=mix_prob
    )
    model.load_state_dict(pretrained["model"])
    model.to(device).eval()

    label = torch.tensor(CarlaDataset.coco_ci_map["dog" # TODO: change label
                                                 ]).unsqueeze(0).to(device)

    x_adv = model.decode(model.forward(x_t, label)) # x_{adv}
    img = render_a_image(neural_renderer, image, x_adv)
    cv2.imwrite("images/test.png", img)

    tt_adv = neural_renderer.textures.clone()
    tt_adv[:, neural_renderer.selected_faces, :] = x_adv
    tt_adv = tt_adv.detach().cpu()

    b, g, r = torch.split(tt_adv, [1, 1, 1], dim=-1) # 交换颜色通道
    tt_adv = torch.cat((r, g, b), dim=-1)

    tt_adv = tt_adv.squeeze(0)
    # tt_adv = torch.tanh(tt_adv)
    save_textures = tt_adv
    print(save_textures.shape)
    # torch.save(save_textures, "images/audi-adv.pt")
    # nr.save_obj(
    #     "images/audi-adv.obj",
    #     neural_renderer.vertices.squeeze(0),
    #     neural_renderer.faces.squeeze(0),
    #     save_textures,
    # )


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
