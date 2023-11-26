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

model_list = ["yolo", "frcnn","yolox","retinanet","ssd"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="temps/render-person/Town05-0722_0959-person")
    parser.add_argument('--model', type=str, default="yolo", choices=model_list)
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
    # yolox_model = init_detector(# YOLOX
    #     'libs/mmdetection/configs/yolox/yolox_l_8xb8-300e_coco.py',
    #     'pretrained/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
    #     device=device
    # )
    yolox_model = init_detector(# YOLOX
        'libs/mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py',
        'pretrained/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
        device=device
    )
    retinanet_model = init_detector(# RetinaNet
        'libs/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py',
        "pretrained/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",device=device)
    ssd_model = init_detector(# SSD
        "libs/mmdetection/configs/ssd/ssd512_coco.py",
        "pretrained/ssd512_coco_20210803_022849-0a47a1ca.pth",device=device)

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
    print(data_dir)
    reuslt_list = []
    attack_names = []
    save_dir = f"{data_dir}-{args.model}"
    
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
            elif args.model in [m for m in model_list if m != "yolo"]:
                if args.model == "frcnn":
                    model= faster_rcnn
                elif args.model == "yolox":
                    model = yolox_model
                elif args.model == "retinanet":
                    model = retinanet_model
                elif args.model == "ssd":
                    model = ssd_model
                else:
                    raise KeyError
                result = inference_detector(model, img)
                pred_results = []
                for [x1, y1, x2, y2], conf, cls in zip(
                    (result.pred_instances.bboxes).cpu().numpy(),
                    (result.pred_instances.scores).cpu().numpy(),
                    (result.pred_instances.labels).cpu().numpy(),
                ):
                    if conf > 0.6:
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
                iou = box_iou(torch.tensor([pbox]), torch.tensor([tbox]))[0].item()
                # print(si, [pbox, pconf, pcls],iou)
                color = (0, 0, 0)

                if (iou > 0.5) and (pcls == car_idx):
                    is_detected = True
                if iou > 0.5:
                    if pcls == car_idx: #
                        # color = (0, 255, 0)
                        color = CarlaDataset.color_map[CarlaDataset.coco_ic_map[pcls]]
                    else:
                        # color = (0, 0, 255)
                        color = CarlaDataset.color_map[CarlaDataset.coco_ic_map[pcls]]

                    enhance_info.append([[x1, y1, x2, y2], pconf, pcls, color])
                else:
                    # color = (0, 0, 0)
                    color = CarlaDataset.color_map[CarlaDataset.coco_ic_map[pcls]]
                    detect_info.append([[x1, y1, x2, y2], pconf, pcls, color])

            detect_result = []
            img_labeled = img.copy()
            for [[x1, y1, x2, y2], pconf, pcls, color] in detect_info:
                pcls_text = f"{CarlaDataset.coco_ic_map[pcls]}:{pconf:.2f}"
                detect_result.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "conf": pconf,
                        "clsanme": CarlaDataset.coco_ic_map[pcls],
                        "clsid": pcls,
                        "color": color
                    }
                )
                cv2.rectangle(img_labeled, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_labeled, pcls_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            for [[x1, y1, x2, y2], pconf, pcls, color] in enhance_info:
                pcls_text = f"{CarlaDataset.coco_ic_map[pcls]}:{pconf:.2f}"
                detect_result.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "conf": pconf,
                        "clsanme": CarlaDataset.coco_ic_map[pcls],
                        "clsid": pcls,
                        "color": color
                    }
                )
                cv2.rectangle(img_labeled, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_labeled, pcls_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            os.makedirs(save_dir, exist_ok=True)
            file_base=os.path.splitext(file)[0]
            cv2.imwrite(f"{save_dir}/{file_base}-labeled.jpg", img_labeled)
            cv2.imwrite(f"{save_dir}/{file_base}.jpg", img)
            with open(f"{save_dir}/{file_base}.json","w")as f:
                json.dump(detect_result,f,indent=4)
            

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
        json.dump({
            "model": args.model,
            "ap": ap_dict,
        }, f, indent=4)


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
