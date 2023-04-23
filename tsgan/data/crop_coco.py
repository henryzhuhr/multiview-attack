import os
import random
from typing import List
from torch.utils import data
from pycocotools.coco import COCO
from PIL import Image
import tqdm
import yaml

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
        load_all_class: bool = False, # rewrite the categories in config file
    ):
        self.data_type = "train" if is_train else "val"
        with open(config_file, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.coco_root = os.path.expanduser(config_dict["coco_root"])
        coco = COCO(f"{self.coco_root}/annotations/instances_{self.data_type}2017.json")

        cats = coco.loadCats(coco.getCatIds())
        COCO_CATEGORIES_MAP = {cat['id']: cat['name'] for cat in cats}
        
        if not load_all_class and "categories" in config_dict:
            categories = {cat['id']: cat['name'] for cat in cats if (cat['name'] in config_dict["categories"])}
        else:
            categories = COCO_CATEGORIES_MAP
        # print('COCO (selected) categories:', len(categories))
        # print(categories)

        # print("load ids", list(categories.keys()))
        img_ids = []
        for id in list(categories.keys()):
            img_ids.extend(coco.getImgIds(catIds=[id]))

        self.COCO_CATEGORIES_MAP = COCO_CATEGORIES_MAP
        self.COCO_CLASS = list(COCO_CATEGORIES_MAP.values())
        self.categories = categories                                                                            # selected categories
        self.transform = transform
        self.object_list = self.prepare_crop(coco, img_ids, self.data_type, categories, min_obj_size)
        print(
            f"[Data] {self.data_type} set get {len(categories)} classes with {len(self.object_list)} objects:",
            categories
        )

    def prepare_crop(self, coco: COCO, img_ids, data_type, categories, min_obj_size):

        object_list = []
        pbar = tqdm.tqdm(img_ids)
        for img_id in pbar:
            img_info = coco.loadImgs(img_id)[0]
            # get img
            file_name = img_info["file_name"]
            pbar.set_description(f"Processing {data_type} {img_id} {file_name} images")
            # get ann
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

            for object in anns:
                iscrowd = object["iscrowd"]
                if iscrowd == 1:
                    continue
                bbox = object["bbox"]
                category_id = object["category_id"]
                if category_id not in categories:
                    continue
                ann_id = object["id"]

                x, y, w, h = bbox
                if (w < min_obj_size) and (h < min_obj_size):
                    continue
                object_crop_info = {
                    "file_name": file_name,
                    "bbox": [x, y, w, h],
                    "category_id": category_id,
                    "category_name": self.COCO_CATEGORIES_MAP[category_id],
                    "predict_id": self.COCO_CLASS.index(self.COCO_CATEGORIES_MAP[category_id])
                }
                object_list.append(object_crop_info)
        return object_list

    def __len__(self):
        return len(self.object_list)

    def __getitem__(self, index: int):

        object_info = self.object_list[index]
        file_name = object_info["file_name"]
        bbox = object_info["bbox"]
        category_id = object_info["category_id"]
        category_name = object_info["category_name"]
        predict_id = object_info["predict_id"]

        img = Image.open(f"{self.coco_root}/{self.data_type}2017/{file_name}").convert('RGB')

        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
        sub_img = img.crop([x1, y1, x2, y2])
        pad_img = pad_image(sub_img, [max(sub_img.size)] * 2)
        # pad_img = pad_img.resize([224, 224])
        if self.transform:
            pad_img = self.transform(pad_img)

        return {
            "image": pad_img,
            "category_id": category_id,
            "category_name": category_name,
            "predict_id": predict_id,
        }

    def class_detail(self):
        class_dict = {}
        for i in range(self.__len__()):
            object_info = self.object_list[i]
            category_name = object_info["category_name"]
            if category_name not in class_dict:
                class_dict[category_name] = 0
            class_dict[category_name] += 1
        return class_dict


if __name__ == "__main__":
    data_set = CroppedCOCO(config_file='configs/coco.yaml', is_train=False)
    os.makedirs("tmp/cropped-coco", exist_ok=True)
    for i, data_obj in enumerate(data_set):
        image = data_obj["image"]
        category_id = data_obj["category_id"]
        print(category_id, COCO_CATEGORIES_MAP[category_id])
        # save_name = f"tmp/cropped-coco/{i}-{category_id}-{data_set.categories[category_id]}.jpg"
        # image.save(save_name)
