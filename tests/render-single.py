from typing import Dict
import os, sys
import json
import argparse

import tqdm
import cv2
import numpy as np

import torch
from torch import Tensor

sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
from modules.data.dataset import CarlaDataset
from modules.render import NeuralRenderer
from modules import types

import neural_renderer as nr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model', type=str, default='data/models/vehicle-YZ.obj')
    parser.add_argument('--ref_image', type=str, default='data/example3_ref.png')
    scence_name = 'Town10HD-point_0000-distance_001-direction_1'
    parser.add_argument('--scence_image', type=str, default=f'data/dataset/images/{scence_name}.png')
    parser.add_argument('--scence_label', type=str, default=f'data/dataset/labels/{scence_name}.json')

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--data_root', type=str, default='tmp/data')
    parser.add_argument('--save_dir', type=str, default='tmp/out')
    return parser.parse_args()


def render_image():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载渲染器
    with open('data/models/selected_faces.txt', 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model, selected_faces=selected_faces, texture_size=4, image_size=800, device=args.device
    )
    neural_renderer.to(neural_renderer.textures.device)
    neural_renderer.renderer.to(neural_renderer.textures.device)
    print('textures: ',neural_renderer.textures.size())

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

    (
        rgb_images,
        depth_images,
        alpha_images,
    ) = neural_renderer.renderer.forward(
        neural_renderer.vertices,
        neural_renderer.faces,
        torch.tanh(neural_renderer.textures),
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

    cv2.imwrite(os.path.join(args.save_dir, f'{name}.png'), render_image)


def render_dataset():
    args = get_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 加载渲染器
    with open('data/models/selected_faces.txt', 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model, selected_faces=selected_faces, texture_size=4, image_size=800, device=args.device
    )
    neural_renderer.to(neural_renderer.textures.device)
    neural_renderer.renderer.to(neural_renderer.textures.device)

    img_ref = cv2.resize(cv2.imread(args.ref_image), [800, 800]).astype('float32')
    image_ref = torch.from_numpy(img_ref / 255.).permute(2, 0, 1).to(args.device)

    dataset = CarlaDataset(args.data_root)

    conduct = None # 'Town01-point_0002-distance_000-direction_5'#
    for index, dataitem in enumerate(dataset):
        image = dataitem['image']
        label = dataitem['label']

        # (vehicle_transform, camera_transform, fov,) = label
        vehicle_transform: types.carla.Transform = label[0]
        camera_transform: types.carla.Transform = label[1]
        fov = label[2]
        name = label[3]

        if conduct is not None:
            if name != conduct:
                continue
        print(name)
        neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)

        optimizer = torch.optim.Adam(neural_renderer.parameters(), lr=0.1, betas=(0.5, 0.999))
        # loop = tqdm.tqdm(range(20))
        for _ in range(20):
            optimizer.zero_grad()
            # neural_renderer.renderer.eye = nr.get_points_from_angles(2.73, 0, np.random.uniform(0, 360))
            out: Tensor = neural_renderer.forward()
            loss = torch.sum((out - image_ref)**2)
            loss.backward()
            optimizer.step()

        add_textures = neural_renderer.textures
        for face_id in neural_renderer.selected_faces:
            add_textures[0, face_id - 1, :, :, :, :] = neural_renderer.render_textures[0, face_id - 1, :, :, :, :]

        (
            rgb_images,
            depth_images,
            alpha_images,
        ) = neural_renderer.renderer.forward(
            neural_renderer.vertices,
            neural_renderer.faces,
            torch.tanh(add_textures),
        )
        rgb_image: Tensor = rgb_images[0]
        rgb_img: np.ndarray = rgb_image.detach().cpu().numpy() * 255
        rgb_img = rgb_img.transpose(1, 2, 0)

        alpha_image: Tensor = alpha_images[0]
        alpha_channel: np.ndarray = alpha_image.detach().cpu().numpy()

        # img_h, img_w = rgb_img.shape[: 2]
        # x_right = 0 # x 方向向右平移
        # y_down = -img_h*0.04 # y 方向向下平移
        # mat_affine = np.float32([[1, 0, x_right], [0, 1, y_down]])
        # rgb_img = cv2.warpAffine(rgb_img, mat_affine, (img_w, img_h))
        # alpha_channel = cv2.warpAffine(alpha_channel, mat_affine, (img_w, img_h))

        render_image = np.zeros(rgb_img.shape)
        for x in range(alpha_channel.shape[0]):
            for y in range(alpha_channel.shape[1]):
                alpha = alpha_channel[x][y]
                render_image[x][y] = alpha * rgb_img[x][y] + (1 - alpha) * image[x][y]

        cv2.imwrite(
            os.path.join(
                args.save_dir,
                               # 'image.png',
                f'{name}.png',
            ),
            image
        )

        print(camera_transform, vehicle_transform, fov)
        ct_mat = f"({camera_transform.location.x}, {camera_transform.location.y}, {camera_transform.location.z})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        render_image = cv2.putText(render_image, ct_mat, (0, 40), font, 1, (255, 255, 255), 2)
        cv2.imwrite(
            os.path.join(
                args.save_dir,
                               # 'render.png',
                f'{name}.png',
            ),
            render_image
        )
        print()
        if conduct is not None:
            if name == conduct:
                break
                               # break
                               # sleep(2)
                               # if index>=20:
                               #     break


if __name__ == '__main__':
    # render_dataset()
    render_image()
