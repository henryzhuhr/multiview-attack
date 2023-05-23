"""
渲染车辆到全部数据集，筛选出渲染歪的
"""
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
    texture_size: int
    data_dir: str
    device: str


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model', type=str, default='assets/vehicle.obj')
    parser.add_argument('--world_map', type=str, default="Town01")
    parser.add_argument('--texture_size', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default="tmp/data-maps")
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def main():
    args: TypeArgs = get_args()
    device = torch.device(args.device)

    # Load Neural Renderer
    neural_renderer = NeuralRenderer(
        args.obj_model,
        texture_size=4,
        image_size=800,
        device=args.device,
    )

    data_set = CarlaDataset(carla_root=f"tmp/data-maps/{args.world_map}", categories=[], is_train=False)
    num_classes = len(data_set.coco_ic_map)

    # --- Load Detector ---
    with open("configs/hyp.scratch-low.yaml", "r") as f:
        hyp: dict = yaml.safe_load(f)
    detector = Model("configs/yolov5s.yaml", ch=3, nc=num_classes, anchors=hyp.get('anchors')).to(device)
    detector.nc = num_classes
    detector.hyp = hyp
    detector_loss = ComputeLoss(detector)
    ckpt = torch.load("pretrained/yolov5s.pt", map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    detector.load_state_dict(csd, strict=False)
    detector.eval()
    conf_thres, iou_thres = 0.25, 0.6


    images_dir = f"tmp/data-maps/{args.world_map}/images"
    labels_dir = f"tmp/data-maps/{args.world_map}/labels"
    print(images_dir, labels_dir)

    carla_label_list = {}
    for i_f, file in enumerate(os.listdir(labels_dir)):
        if file.endswith(".json"):
            label = CarlaDataset.load_carla_label(os.path.join(labels_dir, file))
            corresponding_image_file = os.path.join(images_dir, f"{label['name']}.png")
            if os.path.exists(corresponding_image_file):
                carla_label_list[file] = label

    pbar = tqdm.tqdm(carla_label_list.items())
    for i_d, (file, label) in enumerate(pbar):
        pbar.set_description(f"[{i_d}] {file}")

        image_file = os.path.join(images_dir, f"{label['name']}.png")
        image = torch.from_numpy(cv2.imread(image_file)).permute(2, 0, 1).unsqueeze(0).float() / 255.
        image = image.to(device)
        r_p = {"ct": label["camera_transform"], "vt": label["vehicle_transform"], "fov": label["fov"]}

        render_image, _, _, detect_img = render(neural_renderer,  image, r_p)
        with torch.no_grad():
            eval_pred, _ = detector.forward(render_image) # real
        pred_results = non_max_suppression(eval_pred, conf_thres, iou_thres, None, False)[0]
        w, h = detect_img.shape[: 2]
        bboxes = []

        if len(pred_results):
            for *xyxy, conf, category in pred_results:
                x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                bboxes.append([0, int(category), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
        render_dir = f"tmp/data-maps/{args.world_map}/render"
        os.makedirs(render_dir, exist_ok=True)

        cv2.imwrite(f"{render_dir}/{label['name']}.png", detect_img)

        with open(f"{labels_dir}/{file}", "r") as f:
            label_dict = json.load(f)
        label_dict["bboxes"] = bboxes
        with open(f"{labels_dir}/{file}", "w") as f:
            json.dump(label_dict, f, indent=4)


def render(
    neural_renderer: NeuralRenderer, base_image: torch.Tensor, render_params: dict
):

    neural_renderer.set_render_perspective(render_params["ct"], render_params["vt"], render_params["fov"])
    rgb_image, _, alpha_image = neural_renderer.forward(torch.tanh(neural_renderer.textures))
    render_image = alpha_image * rgb_image + (1 - alpha_image) * base_image
    render_img = np.ascontiguousarray(render_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255)
    return render_image, rgb_image, alpha_image, render_img.astype(np.uint8)


if __name__ == '__main__':
    main()
