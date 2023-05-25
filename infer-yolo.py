import json
import os
import argparse
import datetime
from typing import List
import tqdm
import yaml

import numpy as np
import cv2
import torch

from models.gan import TextureGenerator
from models.render import NeuralRenderer
from models.data.carladataset import CarlaDataset
from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import non_max_suppression

cstr = lambda s: f"\033[01;32m{s}\033[0m"
logt = lambda: "\033[01;32m{%d}\033[0m" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

name = "Town10HD-point_0003-distance_125-direction_1"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model', type=str, default='assets/vehicle.obj')
    parser.add_argument('--selected_faces', type=str, default='assets/faces-vehicle-std.txt')
    parser.add_argument('--scence_image', type=str, default=f"temp/data-maps/Town10HD/images/{name}.png")
    parser.add_argument('--scence_label', type=str, default=f"temp/data-maps/Town10HD/labels/{name}.json")
    # parser.add_argument('--scence_image', type=str, default="images/carla-scene.png")
    # parser.add_argument('--scence_label', type=str, default="images/carla-scene.json")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


class ArgsType:
    object_model: str
    selected_faces: str
    scence_image: str
    scence_label: str
    device: str


def main():
    args: ArgsType = get_args()
    device = args.device

    # --- Load Neural Renderer ---
    with open(args.selected_faces, 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model,
        selected_faces=selected_faces,
        texture_size=4,
        image_size=800,
        device=args.device,
    )

    # --- Load Detector ---
    with open("configs/hyp.scratch-low.yaml", "r") as f:
        hyp: dict = yaml.safe_load(f)
    detector = Model("configs/yolov5s.yaml", ch=3, nc=80, anchors=hyp.get('anchors')).to(device)
    detector.nc = 80
    detector.hyp = hyp
    detector_loss = ComputeLoss(detector)
    ckpt = torch.load("pretrained/yolov5s.pt", map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    detector.load_state_dict(csd, strict=False)
    detector.eval()
    conf_thres, iou_thres = 0.25, 0.6

    # --- Load Image and label ---
    img = cv2.imread(args.scence_image)
    image = (torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.).to(device)
    with open(args.scence_label, 'r') as f:
        label_dict = json.load(f)
        vehicle_transform = CarlaDataset.convert_dict_transform(label_dict['vehicle'])
        camera_transform = CarlaDataset.convert_dict_transform(label_dict['camera'])
        fov = label_dict['camera']['fov']
        name = label_dict['name']
    neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)

    render_clean_image = render_a_image(neural_renderer, image)
    render_adv_image = render_a_image(neural_renderer, image, is_adv=True)
    os.makedirs("temp", exist_ok=True)
    while True:
        render_clean_img, detect_clean_img,pclass = infer_yolo(detector, render_clean_image, conf_thres, iou_thres)

        render_adv_img, detect_adv_img,pclass = infer_yolo(detector, render_adv_image, conf_thres, iou_thres)
        cv2.imwrite("temp/test.png", cv2.vconcat([detect_clean_img, detect_adv_img]))
        if pclass!="car":
            break

    
    


def render_a_image(neural_renderer: NeuralRenderer, base_image: torch.Tensor,is_adv=False):
    x_t = neural_renderer.textures[:, neural_renderer.selected_faces, :]
    tt_adv = neural_renderer.textures
    
    # random_texture = torch.load("temp/random_texture.pt")# TODO
    if is_adv:
        sl_t=torch.rand_like(x_t)
    else:
        sl_t=x_t
    
    tt_adv[:, neural_renderer.selected_faces, :] = sl_t
    rgb_image, _, alpha_image = neural_renderer.forward(torch.tanh(tt_adv))
    render_image = rgb_image #alpha_image * rgb_image + (1 - alpha_image) * base_image
    return render_image


def infer_yolo(detector: Model, render_image: torch.Tensor, conf_thres=0.25, iou_thres=0.6):

    # crop the center 60% of render_image
    w, h = render_image.shape[2 :]
    render_image = render_image[:, :, int(w * 0.2): int(w * 0.8), int(h * 0.2): int(h * 0.8)]

    render_img = np.ascontiguousarray(render_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255)
    render_img = render_img.astype(np.uint8)
    detect_img = render_img.copy()
    with torch.no_grad():
        eval_pred, train_preds = detector.forward(render_image) # real
        pred_results = non_max_suppression(eval_pred, conf_thres, iou_thres, None, False)[0]
    w, h = detect_img.shape[: 2]
    if len(pred_results):
        for *xyxy, conf, category in pred_results:
            pclass = CarlaDataset.coco_ic_map[int(category)]
            text = f'{pclass}:{conf:.2f}'
            x1, y1, x2, y2 = [int(xy) for xy in xyxy]

            color = (0, 255, 0)
            if pclass != "car":
                color = (0, 0, 255)
            cv2.rectangle(detect_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(detect_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return render_img, detect_img, pclass


if __name__ == "__main__":
    main()