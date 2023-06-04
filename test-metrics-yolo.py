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
from utils.general import non_max_suppression

from mmdet.apis import init_detector, inference_detector

cstr = lambda s: f"\033[01;32m{s}\033[0m"
logt = lambda: "\033[01;32m{%d}\033[0m" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

model_list = ["yolo", "frcnn"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="tmp/test/Town01-dog_kite_skateboard-0602_0945")
    parser.add_argument('--model', type=str, default=model_list[0], choices=model_list)
    parser.add_argument('--device', type=str, default="cuda:1")
    args = parser.parse_args()

    return args


class ArgsType:
    data_dir: str
    model: str
    device: str


def main():
    args: ArgsType = get_args()
    device = torch.device(args.device)
    print( f"Test: {args.model}")

    # --- Load Detector ---
    with open("configs/hyp.scratch-low.yaml", "r") as f:
        hyp: dict = yaml.safe_load(f)
    detector = Model("configs/yolov5s.yaml", ch=3, nc=80, anchors=hyp.get('anchors')).to(device)
    detector.nc = 80
    detector.hyp = hyp
    ckpt = torch.load("pretrained/yolov5s.pt", map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    detector.load_state_dict(csd, strict=False)
    detector.eval()
    conf_thres, iou_thres = 0.25, 0.5

    faster_rcnn = init_detector(
        'libs/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
        'pretrained/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        device=device
    )
    # faster_rcnn = init_detector(# YOLOX
    #     'libs/mmdetection/configs/yolox/yolox_tiny_8xb8-300e_coco.py',
    #     'pretrained/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
    #     device=device
    # )

    car_idx = CarlaDataset.coco_ci_map['car']

    data_dir = args.data_dir

    carla_label_list = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), "r") as f:
                label: dict = json.load(f)
                # img_file = os.path.join(data_dir, f"{label['name']}.png")
                # if ('box' in label) and os.path.exists(img_file):
                carla_label_list.append(label)

    pbar = tqdm.tqdm(carla_label_list)
    reuslt_list = []
    attack_names = []
    save_dir = f"{data_dir}-{args.model}"
    os.makedirs(save_dir, exist_ok=True)
    for i_d, label in enumerate(pbar):
        [x1, y1, x2, y2] = label['bbox']
        w, h = label['size']
        [x1, y1, x2, y2] = [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tbox = [x1, y1, x2, y2]

        correct = {}

        for attack_name, file in label['imgs'].items():
            if attack_name not in attack_names:
                attack_names.append(attack_name)
            img_file = os.path.join(data_dir, file)
            img = cv2.imread(img_file)
            image = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.
            image = image.to(device)

            if args.model == "yolo":
                with torch.no_grad():
                    eval_pred, train_preds = detector.forward(image) # real
                    pred_results = non_max_suppression(eval_pred, conf_thres, iou_thres, None, False)[0]
            elif args.model == "frcnn":
                result = inference_detector(faster_rcnn, img)
                pred_results = []
                for [x1, y1, x2, y2], conf, cls in zip(
                    (result.pred_instances.bboxes).cpu().numpy(),
                    (result.pred_instances.scores).cpu().numpy(),
                    (result.pred_instances.labels).cpu().numpy(),
                ):
                    if conf > 0.5:
                        pred_results.append([int(x1), int(y1), int(x2), int(y2), conf, cls])
            else:
                raise NotImplementedError

            is_detected = False
            detect_info = []
            enhance_info = []
            for si, pred in enumerate(pred_results):

                [x1, y1, x2, y2] = [int(x) for x in pred[: 4]]
                pbox = [x1, y1, x2, y2]
                pconf = float(pred[4])
                pcls = int(pred[5])
                pcls_text = f"{CarlaDataset.coco_ic_map[pcls]}:{pconf:.2f}"

                iou = box_iou(torch.tensor([pbox]), torch.tensor([tbox]))[0].item()
                # print(si, [pbox, pconf, pcls],iou)
                color = (0, 0, 0)

                if (iou > 0.5) and (pcls == car_idx):
                    is_detected = True
                if iou > 0.5:
                    if pcls == car_idx: #
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    enhance_info.append([[x1, y1, x2, y2], pcls_text, color])
                else:
                    color = (0, 0, 0)
                    detect_info.append([[x1, y1, x2, y2], pcls_text, color])

            for [[x1, y1, x2, y2], pcls_text, color] in detect_info:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, pcls_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            for [[x1, y1, x2, y2], pcls_text, color] in enhance_info:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, pcls_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imwrite(f"{save_dir}/{file}", img)

            correct[attack_name] = is_detected

        reuslt_list.append(correct)
        pbar.set_description(f"[{i_d}] {correct}")

    result_count = {n: 0 for n in attack_names}

    for item in reuslt_list:
        for k, v in item.items():
            if v:
                result_count[k] += 1

    ap_dict = {k: v / len(reuslt_list) * 100 for k, v in result_count.items()}
    for k, v in ap_dict.items():
        print(f"{k}: {v:.4f}%")

    with open(f"{data_dir}-{args.model}-result.json", "w") as f:
        json.dump(ap_dict, f, indent=4)


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
"""
{
"name": "Town10HD-point_0000-distance_000-direction_1",
"imgs": {
    "clean": "Town10HD-point_0000-distance_000-direction_1-clean.png",
    "noise": "Town10HD-point_0000-distance_000-direction_1-noise.png",
    "pacg-dog": "Town10HD-point_0000-distance_000-direction_1-pacg-dog.png",
    "DAS": "Town10HD-point_0000-distance_000-direction_1-DAS.png",
    "FCA": "Town10HD-point_0000-distance_000-direction_1-FCA.png"
},
"bbox": [
    0.28625,
    0.38875,
    0.65625,
    0.605
],
"size": [
    800,
    800
]
}
"""