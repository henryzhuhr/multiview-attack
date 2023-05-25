import json
from nis import cat
import os
import argparse
import datetime
from time import sleep
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="tmps/detected/audi/Town10HD")
    parser.add_argument('--device', type=str, default="cuda:1")
    args = parser.parse_args()

    return args


class ArgsType:
    data_dir: str
    world_map: str
    pretrained: str
    device: str


def main():
    args: ArgsType = get_args()
    device = torch.device(args.device)

    nowt = datetime.datetime.now().strftime("%m%d%H%M")

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
    conf_thres, iou_thres = 0.25, 0.5
    car_idx = CarlaDataset.coco_ci_map['car']

    data_dir = args.data_dir

    carla_label_list = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), "r") as f:
                label: dict = json.load(f)
                img_file = os.path.join(data_dir, f"{label['name']}.png")
                if ('box' in label) and os.path.exists(img_file):
                    carla_label_list.append(label)

    pbar = tqdm.tqdm(carla_label_list)
    num_detected = 0
    num_total = 0
    for i_d, label in enumerate(pbar):

        image_file = os.path.join(data_dir, f"{label['name']}.png")
        img = cv2.imread(image_file)
        image = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.
        image = image.to(device)

        [x1, y1, x2, y2] = label['box']
        w, h = img.shape[: 2]
        [x1, y1, x2, y2] = [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tbox = [x1, y1, x2, y2]

        with torch.no_grad():
            eval_pred, train_preds = detector.forward(image) # real
            pred_results = non_max_suppression(eval_pred, conf_thres, iou_thres, None, False)[0]

        is_detected = False
        for si, pred in enumerate(pred_results):

            [x1, y1, x2, y2] = [int(x) for x in pred[: 4]]
            pbox = [x1, y1, x2, y2]
            pconf = float(pred[4])
            pcls = int(pred[5])

            iou = box_iou(torch.tensor([pbox]), torch.tensor([tbox]))[0].item()
            # print(si, [pbox, pconf, pcls],iou)
            if (iou > 0.5) and (pcls == car_idx):
                is_detected = True

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, f"{CarlaDataset.coco_ic_map[pcls]}:{pconf:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2
            )

        # cv2.imwrite('temp/test.png', img)
        # sleep(1)
        # break
        num_total += 1
        if is_detected:
            num_detected += 1

        # pbar.set_postfix_str(f"detected: {num_detected}/{num_total}")
        ap = num_detected / num_total * 100
        pbar.set_description(f"[{ap:.4f}%] [{i_d}] {file} ")

    print(f"AP: {num_detected / num_total * 100}%")


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


if __name__ == "__main__":
    main()