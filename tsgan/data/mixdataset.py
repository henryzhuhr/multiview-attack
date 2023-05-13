import json
import os
import random
from typing import Dict, List, Union
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

cstri = lambda s: f"\033[01;32m{s}\033[0m"
cstrs = lambda s: f"\033[01;36m{s}\033[0m"
cstrw = lambda s: f"\033[01;33m{s}\033[0m"

# class CroppedCOCOCarlaMixDatasetSampler(data.Sampler):

COCO_CATEGORIES_MAP = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}


class CarlaDatasetDir:
    def __init__(self, root, data_type: str = None) -> None:
        base_root = os.path.join(root, data_type) if data_type is not None else root
        self.images_dir = os.path.join(base_root, "scenes")
        self.labels_dir = os.path.join(base_root, "labels")
        self.segmentations_dir = os.path.join(base_root, "segmentations")


def load_carla_label(label_path):
    def convert_dict_transform(td: Dict):
        # td: transform_dict
        return types.carla.Transform(
            location=types.carla.Location(x=td['location']['x'], y=td['location']['y'], z=td['location']['z']),
            rotation=types.carla.Rotation(pitch=td['rotation']['pitch'], yaw=td['rotation']['yaw'], roll=td['rotation']['roll']
        ))  # yapf:disable

    with open(label_path, "r") as f:
        label_dict = json.load(f)
    vehicle_transform = convert_dict_transform(label_dict["vehicle"])
    camera_transform = convert_dict_transform(label_dict["camera"])
    fov = label_dict["camera"]["fov"]
    name = label_dict["name"]
    return {"name": name, "vehicle_transform": vehicle_transform, "camera_transform": camera_transform, "fov": fov}


class CarlaDataset(data.Dataset):
    def __init__(self, carla_root: str, categories: Union[str, List[str]]):

        carla_label_list = []
        carla_dir = CarlaDatasetDir(carla_root)
        for i_f, file in enumerate(os.listdir(carla_dir.labels_dir)):
            if file.endswith(".json"):
                label = load_carla_label(os.path.join(carla_dir.labels_dir, file))
                corresponding_image_file = os.path.join(carla_dir.images_dir, f"{label['name']}.png")
                if os.path.exists(corresponding_image_file):
                    carla_label_list.append(label)

        coco_category_index = {cn: idx for idx, cn in enumerate(COCO_CATEGORIES_MAP.values())}
        coco_index_category = {idx: cn for idx, cn in enumerate(COCO_CATEGORIES_MAP.values())}
        if isinstance(categories, str):
            categories_list = [coco_category_index[categories]]
        elif isinstance(categories, list):
            categories_list = [coco_category_index[c] for c in categories]
        else:
            raise TypeError

        self.carla_dir = carla_dir
        self.carla_label_list = carla_label_list
        self.categories_list = categories_list
        self.coco_ic_map = coco_category_index # {"class_name":index}
        self.coco_ci_map = coco_index_category # {"index":class_name}

    def __len__(self):
        return self.carla_label_list.__len__()

    def __getitem__(self, index: int):
        # carla
        # item = random.choice(self.carla_label_list)
        item = self.carla_label_list[index]
        image_file = os.path.join(self.carla_dir.images_dir, f"{item['name']}.png")
        image = cv2.imread(image_file)
        item["image"] = image
        item["label"] = random.choice(self.categories_list)
        return item


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

        carla_dir = CarlaDatasetDir(os.path.expanduser(carla_config["root"]))
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
        carla_sample_obj = random.choice(self.carla_label_list)
        # carla_sample_obj = self.carla_label_list[0]

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