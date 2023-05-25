import abc
from time import sleep
from typing import Dict, List
import os, sys
import json
import argparse
import cv2
import numpy as np

import torch
import tqdm
import yaml
from models.data.carladataset import CarlaDataset
from models.render import NeuralRenderer
from models.data import types
import neural_renderer as nr

from models.yolo import Model
from utils.general import non_max_suppression
from utils.loss import ComputeLoss


class TypeArgs(metaclass=abc.ABCMeta):
    obj_model: str
    world_map: str
    save_dir: str
    texture_size: int
    data_dir: str
    device: str


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model', type=str, default='assets/audi.obj')
    parser.add_argument('--data_dir', type=str, default="temp/data-maps")
    parser.add_argument('--world_map', type=str, default="Town10HD")
    parser.add_argument('--save_dir', type=str, default="tmps/detected/audi")
    parser.add_argument('--texture_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def main():
    args: TypeArgs = get_args()
    device = torch.device(args.device)

    # --- Load Neural Renderer ---
    neural_renderer = NeuralRenderer(
        args.obj_model,
        texture_size=4,
        image_size=800,
        device=args.device,
    )

    # --- Load Dataset ---
    print(f"{args.data_dir}/{args.world_map}")
    data_set = CarlaDataset(carla_root=f"{args.data_dir}/{args.world_map}", categories=[], is_train=False)
    num_classes = len(data_set.coco_ic_map)

    pbar = tqdm.tqdm(data_set)
    for i_d, item in enumerate(pbar):
        pbar.set_description(f"Processing {i_d}/{len(data_set)}")
        image = item["image"].to(device)
        name = item["name"]
        file = item["file"]
        r_p = {"ct": item["ct"], "vt": item["vt"], "fov": item["fov"]}

        [x1, y1, x2, y2], w, h = get_bbox(neural_renderer, r_p)
        img = btensor2img(image)
        detect_img=cv2.rectangle(img.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite("temp/test.png", detect_img)
        


def get_bbox(neural_renderer: NeuralRenderer, render_params: dict):
    """ 获取边界框 """
    neural_renderer.set_render_perspective(render_params["ct"], render_params["vt"], render_params["fov"])
    _, _, alpha_image = neural_renderer.forward(torch.tanh(neural_renderer.textures))
    binary = np.ascontiguousarray(alpha_image.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    find_boxes = []
    for c in contours:
        [x, y, w, h] = cv2.boundingRect(c)
        find_boxes.append([x, y, x + w, y + h])
    fc = np.array(find_boxes)
    box = [min(fc[:, 0]), min(fc[:, 1]), max(fc[:, 2]), max(fc[:, 3])] # [x1,y1,x2,y2]
    [x1, y1, x2, y2] = [int(b) for b in box]
    return [x1, y1, x2, y2], w, h


def btensor2img(image:torch.Tensor):
    """ batch tensor [1,C,W,H] to image [W,H,C]"""
    return np.ascontiguousarray(image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255)


if __name__ == '__main__':
    main()
