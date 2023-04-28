import argparse
import os
import numpy as np

import torch
import tqdm
import yaml

from models.yolo.yolo import DetectionModel
from tsgan.models.yolo.utils.dataloaders import create_dataloader
from tsgan.models.yolo.utils.general import check_img_size, colorstr, non_max_suppression, scale_boxes, xywh2xyxy
from tsgan.models.yolo.utils.loss import ComputeLoss
from tsgan.models.yolo.utils.metrics import ap_per_class, box_iou



def parse_opt():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--data', type=str, default='data/coco128.yaml')
    # model
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml')
    parser.add_argument('--weights', type=str, default='pretrained/yolov5s.pt')
    parser.add_argument('--hyp', type=str, default='./data/hyps/hyp.scratch-low.yaml')

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=0)
    opt = parser.parse_args()
    return opt


def main(
    data: str,
    cfg: str,
    weights: str,
    hyp: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    workers: int,
):
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    with open(data, "r") as f:
        data_dict: dict = yaml.safe_load(f) # dictionary
    with open(hyp, "r") as f:
        hyp: dict = yaml.safe_load(f)       # load hyps dict

    nc = len(data_dict['names'])
    model = DetectionModel(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device) # create
    model.nc = nc                                                          # attach number of classes to model
    model.hyp = hyp                                                        # attach hyperparameters to model
    compute_loss = ComputeLoss(model)                                      # init loss class
    if weights:
        ckpt = torch.load(weights, map_location='cpu')
        csd = ckpt['model'].float().state_dict()                           # checkpoint state_dict as FP32
        model.load_state_dict(csd, strict=False)                           # load

    gs = max(int(model.stride.max()), 32) # grid size (max stride)

    iouv = torch.linspace(0.5, 0.95, 10, device=device) # iou vector for mAP@0.5:0.95
    niou = iouv.numel()


    train_loader, dataset = create_dataloader(data_dict['train'],
                                              imgsz,
                                              batch_size,
                                              gs,
                                              False,
                                              hyp=hyp,
                                              augment=True,
                                              cache="ram",
                                              rect=False,
                                              rank=-1,
                                              workers=workers,
                                              quad=False,
                                              prefix=colorstr('train: '),
                                              shuffle=True,)

    # eval train diffierent :https://github.com/ultralytics/yolov5/issues/9318#issuecomment-1239193067
    model.train()

    pbar = tqdm.tqdm(train_loader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}') # progress bar
    for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
        print(targets.size(),targets[:,0])
        exit()
        # Inference
        
        imgs=imgs.float()/255
        # with torch.no_grad():
        pred = model.forward(imgs)

        # compute_loss.__call__/return: (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
        loss, loss_items = compute_loss.__call__(pred, targets.to(device)) 
        loss.backward()

if __name__ == '__main__':
    opt = parse_opt()
    main(**vars(opt))