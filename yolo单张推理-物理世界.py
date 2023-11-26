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

name = "IMG_4826.jpg"
def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    img=cv2.imread(name)
    img=cv2.resize(img,(800,800))
    image=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()/255
    image=image.to(device)
    
    eval_pred, train_preds = detector.forward(image) # real
    pred_results = non_max_suppression(eval_pred, conf_thres, iou_thres, None, False)[0]
    for *xyxy, conf, category in pred_results:
        pclass = CarlaDataset.coco_ic_map[int(category)]
        text = f'{pclass}:{conf:.2f}'
        x1, y1, x2, y2 = [int(xy) for xy in xyxy]

        color = (0, 255, 0)
        if pclass != "car":
            color = (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imwrite("IMG.png",img)



if __name__ == "__main__":
    main()