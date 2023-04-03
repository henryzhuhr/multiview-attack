import json
import os
import random
from typing import List
import yaml
import tqdm

from pycocotools.coco import COCO

from PIL import Image, ImageDraw

COCO_MAP={
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    7: 'train',
    8: 'truck',
    9: 'boat',
    15: 'bench',
    16: 'bird',
    19: 'horse',
    20: 'sheep',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    31: 'handbag',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    38: 'kite',
    42: 'surfboard',
    44: 'bottle',
    48: 'fork',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    59: 'pizza',
    60: 'donut',
    62: 'chair',
    65: 'bed',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    90: 'toothbrush'
}


def main(config_file: str):
    with open(config_file, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        print(config_dict)
    coco_root = config_dict["coco_root"]
    selected_categories = config_dict["categories"]

    for data_type in [
                                                         # "train",
        "val",
    ]:
        object_list = process_coco(coco_root, data_type, selected_categories)
        print(f"get {len(object_list)} objects")
        with open(f"configs/cropped_coco-{data_type}.json", "w") as f:
            json.dump(
                {
                    "categories": selected_categories,
                    "coco_root": coco_root,
                    "object_list": object_list,
                },
                f,
                indent=4,
            )


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


def process_coco(
    coco_root: str,
    data_type: str,
    categories: List[str],
    min_obj_size: int = 100,
):
    coco = COCO(f"{coco_root}/annotations/instances_{data_type}2017.json")
    cats = coco.loadCats(coco.getCatIds())
    categories_dict = {cat['id']: cat['name'] for cat in cats if (cat['name'] in categories)}
    print(categories_dict)

    img_ids = []
    for id in list(categories_dict.keys()):
        img_ids.extend(coco.getImgIds(catIds=[id]))
    random.shuffle(img_ids)

    object_list = []
    for img_id in img_ids:
        img_info = coco.loadImgs(ids=img_id)[0]
        # get img
        file_name = img_info["file_name"]
        # pbar.set_description(f"Processing {data_type} {img_id} {file_name} images")
        img = Image.open(f"{coco_root}/{data_type}2017/{file_name}").convert('RGB')
        draw = ImageDraw.Draw(img)
        # get ann
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        for i_obj, object in enumerate(anns):
            
            iscrowd = object["iscrowd"]
            if iscrowd == 1:
                continue
            bbox = object["bbox"]
            category_id = object["category_id"]
            if category_id not in categories_dict:
                continue
            ann_id = object["id"]

            x, y, w, h = bbox
            if (w < min_obj_size) and (h < min_obj_size):
                continue
            print(bbox,category_id,ann_id,COCO_MAP[category_id])
            object_crop_info = {
                "file_name": file_name,
                "bbox": [x, y, w, h],
                "category_id": category_id,
                "category_name": COCO_MAP[category_id],
            }
            object_list.append(object_crop_info)
            x1, y1, x2, y2 = x, y, int(x + w), int(y + h)

            sub_img = img.crop([x1, y1, x2, y2])
            if (sub_img.size[0] < min_obj_size) and (sub_img.size[1] < min_obj_size):
                continue
            sub_img = pad_image(sub_img, [max(sub_img.size)] * 2)
            sub_img = sub_img.resize([224, 224])

            draw.rectangle((x1, y1, x2, y2))
            draw.text((x1, y1), COCO_MAP[object["category_id"]])
            
            
            sub_img.save(f"tmp/cropped-coco/tmp-crop-{i_obj}.jpg")
        img.save(f"tmp/cropped-coco/tmp.jpg")
        exit()
    return object_list


if __name__ == "__main__":
    os.makedirs("tmp/cropped-coco", exist_ok=True)
    main(config_file="configs/coco.yaml")
