import json
import os, sys
import shutil
from random import randint
from pprint import pprint
from typing import Dict, List
import cv2
from torch.utils import data
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import tqdm
import yaml


def pad_image(image: Image.Image, target_size: List[int]):
    iw, ih = image.size         # 原始图像的尺寸
    w, h = target_size          # 目标图像的尺寸
    scale = min(w / iw, h / ih) # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)          # 缩小图像
    new_image = Image.new('RGB', target_size, (0, 0, 0))   # 生成黑色图像
                                                           # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2)) # 将图像填充为中间图像，两侧为灰色的样式

    return new_image


class CroppedCOCO(data.Dataset):
    def __init__(
        self,
        config_file: str,
        is_train: bool = False,
        min_obj_size: int = 100,
        transform=None,
    ):
        self.data_type = "train" if is_train else "val"
        with open(config_file, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.coco_root = config_dict["coco_root"]
        coco = COCO(f"{self.coco_root}/annotations/instances_{self.data_type}2017.json")

        cats = coco.loadCats(coco.getCatIds())
        categories = {cat['id']: cat['name'] for cat in cats if (cat['name'] in config_dict["categories"])}
        # print('COCO (selected) categories:', len(categories))
        # print(categories)

        # print("load ids", list(categories.keys()))
        img_ids = []
        for id in list(categories.keys()):
            img_ids.extend(coco.getImgIds(catIds=[id]))

        self.coco = coco
        self.categories = categories # selected categories
        self.img_ids = img_ids
        self.min_obj_size = min_obj_size
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index: int):
        coco = self.coco

        for img_id in self.img_ids[index :] + self.img_ids[: index]:

            img_info = coco.loadImgs(ids=img_id)[0]
            # get img
            file_name = img_info["file_name"]
            img = Image.open(f"{self.coco_root}/{self.data_type}2017/{file_name}").convert('RGB')
            # get ann
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            for object in anns:
                iscrowd = object["iscrowd"]
                if iscrowd == 0:
                    continue
                bbox = object["bbox"]
                category_id = object["category_id"]
                if category_id not in self.categories:
                    continue
                ann_id = object["id"]

                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
                sub_img = img.crop([x1, y1, x2, y2])
                if (sub_img.size[0] < self.min_obj_size) and (sub_img.size[1] < self.min_obj_size):
                    continue
                sub_img = pad_image(sub_img, [max(sub_img.size)] * 2)
                sub_img = sub_img.resize([224, 224])
                # return sub_img,
                if self.transform:
                    sub_img = self.transform(sub_img)
                return sub_img, category_id
        return None


if __name__ == "__main__":
    data_set = CroppedCOCO("data/coco.yaml")
    os.makedirs("tmp/cropped-coco", exist_ok=True)
    pbar=tqdm.tqdm(data_set)
    for i, (image, label) in enumerate(pbar):
        # image, label = data_set[randint(0,len(data_set.img_ids))]
        save_name = f"tmp/cropped-coco/{i}-{label}-{data_set.categories[label]}.jpg"
        # image.save(save_name)
