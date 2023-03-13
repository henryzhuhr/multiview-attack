from typing import Dict, List
import os, sys
import json
import argparse

import tqdm
import cv2
import numpy as np

import torch
from torch import Tensor, optim, nn

sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
from modules.models.encoder import TextureEncoder
from modules.models.decoder import SimpleDecoder
from modules.data.dataset import CarlaDataset
from modules.render import NeuralRenderer
from modules.loss.ssim import ssim, ms_ssim
from modules import types

import neural_renderer as nr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model', type=str, default='data/models/vehicle-YZ.obj')
    scence_name = 'Town10HD-point_0000-distance_001-direction_1'
    parser.add_argument('--scence_image', type=str, default=f'data/dataset/images/{scence_name}.png')
    parser.add_argument('--scence_label', type=str, default=f'data/dataset/labels/{scence_name}.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='tmp/ssim_l1-0_05')

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--milestones', type=List[int], default=[10000,20000])
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
    encoder = TextureEncoder(num_feature=4, zdim=512)
    encoder.to(args.device)
    decoder = model = SimpleDecoder(
        latent_dim=512,
        num_points=12306,
        texture_size=4,
    )
    loss_func = nn.L1Loss()
    optimizer = optim.SGD(
        [{
            'params': encoder.parameters()
        }, {
            'params': decoder.parameters()
        }],
        lr=1e-1,
    )
    lr_scheduler=optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1)

    encoder.to(args.device)
    decoder.to(args.device)
    loss_func.to(args.device)

    encoder.train()
    decoder.train()
    pabr = tqdm.tqdm(range(10000 * 100))
    for i in pabr:

        x = neural_renderer.textures
        optimizer.zero_grad()
        latent, _ = encoder.forward(x)
        new_x = decoder.forward(latent)

        loss = loss_func.forward(x, new_x)
        expoch_loss = loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        pabr.set_description("iter:%d loss:%.8f" % (i, expoch_loss))

        if i % 10000 == 0:
            print("iter:%d loss:%.4f" % (i, expoch_loss))
            (
                rgb_images,
                depth_images,
                alpha_images,
            ) = neural_renderer.renderer.forward(
                neural_renderer.vertices,
                neural_renderer.faces,
                torch.tanh(new_x),
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
            cv2.imwrite(os.path.join(args.save_dir, f'{name}-{i}.png'), render_image)
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f'{name}-{i}.pt'),
            )

    # ==========================


if __name__ == '__main__':
    render_image()
