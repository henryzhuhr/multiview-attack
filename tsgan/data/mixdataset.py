import json
import os
import random
from typing import Dict, List
import warnings
import cv2
import numpy as np
import torch

from torch.utils import data
from pycocotools.coco import COCO
from PIL import Image
import tqdm
from . import types
import yaml

from .crop_coco import CroppedCOCO
from .utils import CarlaDatasetDir
from .utils import load_carla_label

cstri = lambda s: f"\033[01;32m{s}\033[0m"
cstrs = lambda s: f"\033[01;36m{s}\033[0m"
cstrw = lambda s: f"\033[01;33m{s}\033[0m"

# class CroppedCOCOCarlaMixDatasetSampler(data.Sampler):


class CroppedCOCOCarlaMixDataset(CroppedCOCO):
    def __init__(
        self,
        config_file: str,
        is_train: bool = False,
        min_obj_size: int = 100,
        transform=None,
        load_all_class: bool = False, # for COCO rewrite the categories in config file
        show_detail=False,
    ):
        with open(config_file, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            coco_config = config_dict["coco"]
            carla_config = config_dict["carla"]

        super().__init__(
            config_dict,
            is_train=is_train,
            min_obj_size=min_obj_size,
            transform=transform,
            load_all_class=load_all_class,
            show_detail=show_detail,
        )
        # self.COCO_CATEGORIES_MAP = self.COCO_CATEGORIES_MAP
        # self.COCO_CLASS = self.COCO_CLASS
        # self.categories = self.categories
        # self.transform = self.transform
        # self.object_list = self.object_list

        self.data_type = "train" if is_train else "val"

        # load carla dataset

        carla_dir = CarlaDatasetDir(os.path.expanduser(carla_config["root"]),
                                                                              # self.data_type,
                                   )
        carla_label_list = []
                                                                              # pbar= tqdm.tqdm(os.listdir(carla_dir.labels_dir))
        for i_f, file in enumerate(os.listdir(carla_dir.labels_dir)):
            if file.endswith(".json"):
                label_path = os.path.join(carla_dir.labels_dir, file)
                label = load_carla_label(label_path)
                corresponding_image_file = os.path.join(carla_dir.images_dir, f"{label['name']}.png")
                if os.path.exists(corresponding_image_file):
                    carla_label_list.append(label)
                                                                              #     info= f"{cstri('[Info]')} load {cstrs(i_f)} carla label {cstrs(label['name'])}"
                                                                              # else:
                                                                              #     info= f"{cstrw('[Warning]')} file {cstrw(corresponding_image_file)} not exists, {cstrw('skip it')}"
                                                                              # print(i_f,info)
        self.carla_dir = carla_dir
        self.carla_label_list = carla_label_list

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index: int):
        coco_sample = super().__getitem__(index)
        # carla
        # carla_sample_obj = random.choice(self.carla_label_list)
        carla_sample_obj = self.carla_label_list[index % len(self.carla_label_list)]

        img_file = os.path.join(self.carla_dir.images_dir, f"{carla_sample_obj['name']}.png")
        assert os.path.exists(img_file), f"image file {img_file} not exists"
        carla_image = cv2.imread(img_file)
        carla_sample_obj["image"] = carla_image
        return {
            "coco": coco_sample,
            "carla": carla_sample_obj,
        }

    @staticmethod
    def collate_fn(batch):

        data_list = {
            "coco": {
                "image": [],
                "category_id": [],
                "category_name": [],
                "predict_id": [],
            },
            "carla": {
                "name": [],
                "image": [],
                "render_param": [],
            }
        }
        for item in batch:
            data_list["coco"]["image"].append(item["coco"]["image"])
            data_list["coco"]["category_id"].append(item["coco"]["category_id"])
            data_list["coco"]["category_name"].append(item["coco"]["category_name"])
            data_list["coco"]["predict_id"].append(torch.tensor(item["coco"]["predict_id"]))
            data_list["carla"]["name"].append(item["carla"]["name"])
            data_list["carla"]["image"].append(item["carla"]["image"])
            data_list["carla"]["render_param"].append(
                {
                    "vehicle_transform": item["carla"]["vehicle_transform"],
                    "camera_transform": item["carla"]["camera_transform"],
                    "fov": item["carla"]["fov"],
                }
            )

            # print("\033[01;32m", item, "\033[0m")
            # print()

        data_list["coco"]["image"] = torch.stack(data_list["coco"]["image"])
        data_list["coco"]["predict_id"] = torch.stack(data_list["coco"]["predict_id"], dim=0)

        return data_list