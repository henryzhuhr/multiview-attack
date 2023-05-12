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

import tsgan
from tsgan.render import NeuralRenderer
from tsgan import types
from models.gan import TextureGenerator


class TypeArgs(metaclass=abc.ABCMeta):
    __metaclass__ = abc.ABCMeta
    obj_model: str
    texture_size: int
    latent_dim: int
    scence_image: str
    scence_label: str

    epoches: int
    lr: float
    milestones: List[int]
    device: str

    save_dir: str
    save_name: str
    save_interval: int
    pretained: str


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model', type=str, default='data/models/vehicle-YZ.obj')
    parser.add_argument('--selected_faces', type=str, default='data/models/faces-std.txt')
    parser.add_argument('--texture_size', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--scence_image', type=str, default="images/carla-scene.png")
    parser.add_argument('--scence_label', type=str, default="images/carla-scene.json")

    parser.add_argument('--epoches', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--milestones', type=List[int], default=[8000, 60000])
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='tmp/autoencoder')
    parser.add_argument('--save_name', type=str, default='autoencoder')

    parser.add_argument('--pretained', type=str, default=None) #"tmp/nAE/autoencoder.pt")
    return parser.parse_args()


def main():
    args: TypeArgs = get_args()
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
    ts = args.texture_size
    with open(args.selected_faces, "r") as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model, selected_faces=selected_faces, texture_size=ts, image_size=800, device=args.device
    )
    neural_renderer.to(neural_renderer.textures.device)
    neural_renderer.renderer.to(neural_renderer.textures.device)
    neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)

    x_t = neural_renderer.textures[:, selected_faces, :]
    x_t = x_t.repeat(bs := 8, 1, 1, 1, 1, 1)

    nt = x_t.shape[1]
    print('textures num:%d (%d selected)' % (neural_renderer.textures.shape[1], nt))

    # Load Model
    model = TextureGenerator(
        nt=nt,
        ts=args.texture_size,
        style_dim=args.latent_dim,
    ).cuda().train()
    optimizer = optim.SGD(
        [{
            'params': model.encoder.parameters(),
        }, {
            'params': model.decoder.parameters(),
        }], lr=args.lr
    )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1)

    model.train()
    pabr = tqdm.tqdm(range(args.epoches))
    iter_loss = 0.
    iter_num = 0
    for iter in pabr:
        # Add noise to texture, range [l,h]
        nrotio = np.random.uniform(0.5,1.0)
        # nrotio = 0.5
        x_in = (1 - nrotio) * x_t + nrotio * torch.rand_like(x_t)

        optimizer.zero_grad()
        # --- forward ---
        latent_x = model.encode(x_in)
        x_rec = model.decode(latent_x)
        loss = F.l1_loss(x_t, x_rec)

        iter_loss += loss.item()
        iter_num += 1
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        pabr.set_description(
            f"iter:{iter} "
            f"lr:{optimizer.state_dict()['param_groups'][0]['lr']:.4f} "
            f"loss:{loss.item():.4f} "
            f"nrotio:{nrotio:.4f} "
            f"npoint:{nt}"
        )

        if iter % args.save_interval == 0:
            print(
                f"iter:{iter}", f"arg_loss:{(iter_loss / iter_num):.4f}", f"nrotio:{nrotio:.4f}", f"npoint:{nt}", "\n"
            )
            iter_loss = 0.0
            iter_num = 0
            merged_img = cv2.hconcat(
                [render_a_image(neural_renderer, image, X) for X in (x_in[0].squeeze(0), x_rec[0].squeeze(0))]
            )

            cv2.imwrite(os.path.join(args.save_dir, f'{args.save_name}.png'), merged_img)
            cv2.imwrite(os.path.join(args.save_dir, f'{args.save_name}-{iter}.png'), merged_img)

            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.save_name}-{iter}.pt'))
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.save_name}.pt'))


def render_a_image(neural_renderer: NeuralRenderer, image: cv2.Mat, x_t: Tensor):
    # x_t_full = torch.zeros_like(neural_renderer.textures)
    x_t_full = neural_renderer.textures
    x_t_full[:, neural_renderer.selected_faces, :] = x_t

    rgb_images, _, alpha_images = neural_renderer.renderer.forward(
        neural_renderer.vertices, neural_renderer.faces, torch.tanh(x_t_full)
    )
    rgb_img: cv2.Mat = (rgb_images[0].detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    alpha_channel: cv2.Mat = alpha_images[0].detach().cpu().numpy()

    render_image = np.zeros(rgb_img.shape)
    for x in range(alpha_channel.shape[0]):
        for y in range(alpha_channel.shape[1]):
            alpha = alpha_channel[x][y]
            render_image[x][y] = alpha * rgb_img[x][y] + (1 - alpha) * image[x][y]
    return render_image


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


if __name__ == '__main__':
    main()
