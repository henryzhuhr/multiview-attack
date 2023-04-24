import math
from typing import Dict, List
import os, sys
import json
import argparse

import tqdm
import cv2
import numpy as np

import torch
from torch import Tensor, optim, nn

import tsgan
from tsgan.render import NeuralRenderer
from tsgan import types
import neural_renderer as nr
from torch.utils import data
from torchvision import transforms, utils

from tsgan.models.classifer import resnet50


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--save_dir', type=str, default='tmp/classifier')
    parser.add_argument('--save_name', type=str, default='resnet50')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epoches', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--milestones', type=List[int], default=[3, 10])
    return parser.parse_args()


def main():
    args = get_args()
    args.size = 224

    device = args.device

    os.makedirs(args.save_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
        ]
    )

    croppedcoco_train_set = tsgan.data.CroppedCOCO(
        config_file='configs/coco.yaml',
        is_train=True,
        transform=transform,
        load_all_class=True,
    )
    train_loader = data.DataLoader(
        croppedcoco_train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    croppedcoco_valid_set = tsgan.data.CroppedCOCO(
        config_file='configs/coco.yaml',
        is_train=False,
        transform=transform,
        load_all_class=True,
    )
    valid_loader = data.DataLoader(
        croppedcoco_valid_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )

    # Load Model
    model = resnet50()
    model.load_state_dict(torch.load("pretrained/resnet50-11ad3fa6.pth", map_location="cpu"))
    model.fc = nn.Linear(model.fc.in_features, croppedcoco_train_set.categories.__len__())
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1)

    best_acc = 0
    for epoch in range(args.epoches):
        print('\033[32m', end='')
        print('[Epoch]%d/%d' % (epoch, args.epoches), end='  ')
        print('[Batch Size]%d' % (args.batch_size), end='  ')
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('[LR]%f' % (current_lr), end='  ')
        print(':%s' % (device), end='  ')
        print('\033[0m')

        # ====== train ======
        epoch_loss, epoch_acc = 0., 0. # epoch
        epoch_num = 0                  # num of images trained in an epoch
        model.train()
        pbar = tqdm.tqdm(train_loader)
        for batch_data in pbar:
            images = batch_data["image"].to(device)
            labels = batch_data["predict_id"].to(device)
            epoch_num += images.size(0)

            logits = model.forward(images)

            loss: Tensor = criterion(logits, labels)
            epoch_loss += loss.item() * images.size(0)

            acc = torch.sum(torch.max(logits, 1).indices == labels).item()
            epoch_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            minibatch_log_dict = {
                'loss': loss.item(),
                'acc': acc / images.size(0),
            }
            minibatch_log = ['%s:%.4f' % (key, minibatch_log_dict[key]) for key in minibatch_log_dict.keys()]
            pbar.set_description('Train ' + ' '.join(minibatch_log))
        train_loss = epoch_loss / epoch_num
        train_acc = epoch_acc / epoch_num

        model.eval()
        epoch_loss, epoch_acc = 0., 0. # epoch
        epoch_num = 0                  # num of images trained in an epoch
        pbar = tqdm.tqdm(valid_loader)
        for batch_data in pbar:
            images = batch_data["image"].to(device)
            labels = batch_data["predict_id"].to(device)
            epoch_num += images.size(0)

            with torch.no_grad():
                logits = model.forward(images)

            loss: Tensor = criterion(logits, labels)
            epoch_loss += loss.item() * images.size(0)

            acc = torch.sum(torch.max(logits, 1).indices == labels).item()
            epoch_acc += acc

            minibatch_log_dict = {
                'loss': loss.item(),
                'acc': acc / images.size(0),
            }
            minibatch_log = ['%s:%.4f' % (key, minibatch_log_dict[key]) for key in minibatch_log_dict.keys()]
            pbar.set_description('Valid ' + ' '.join(minibatch_log))

        valid_loss = epoch_loss / epoch_num
        valid_acc = epoch_acc / epoch_num

        print(
            f"{epoch}: train loss:{train_loss:.4f} train acc:{train_acc:.4f}, valid loss:{valid_loss:.4f} valid acc:{valid_acc:.4f}"
        )

        torch.save(
            model.state_dict(),
            f"{args.save_dir}/{args.save_name}-{epoch}.pt",
        )
        if valid_acc > best_acc:
            best_acc = valid_acc

            torch.save(
                model.state_dict(),
                f"{args.save_dir}/{args.save_name}.pt",
            )

        if best_acc > 0.9980:
            break


if __name__ == '__main__':
    main()
