import json
import os, sys
from typing import Dict
import cv2
from torch.utils import data
from PIL import Image

sys.path.append(os.getcwd())
from modules import types


class DatasetDir:
    images = 'images'
    labels = 'labels'
    segmentations = 'segmentations'


def convert_dict_transform(transform_dict: Dict):
    return types.carla.Transform(
        location=types.carla.Location(
            x=transform_dict['location']['x'], y=transform_dict['location']['y'], z=transform_dict['location']['z']
        ),
        rotation=types.carla.Rotation(
            pitch=transform_dict['rotation']['pitch'],
            yaw=transform_dict['rotation']['yaw'],
            roll=transform_dict['rotation']['roll']
        )
    )


class CarlaDataset(data.Dataset):
    def __init__(self, data_root):

        image_files = []
        labels = []

        labels_dir = os.path.join(data_root, 'labels')
        images_dir = os.path.join(data_root, 'images')
        for file in os.listdir(labels_dir):
            if file.endswith('.json'):
                with open(os.path.join(labels_dir, file), 'r') as f:
                    label_dict = json.load(f)
                    state = label_dict['state']
                    if not state:
                        continue
                    image_path = os.path.join(images_dir, label_dict['image'])
                    if os.path.exists(image_path):
                        image_files.append(image_path)
                        vehicle_transform = convert_dict_transform(label_dict['vehicle'])
                        camera_transform = convert_dict_transform(label_dict['camera'])
                        fov = label_dict['camera']['fov']
                        name = label_dict['name']
                        labels.append([vehicle_transform, camera_transform, fov, name])
                    else:
                        raise FileNotFoundError(image_path)
        self.image_files = image_files
        self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return {
            "image": cv2.imread(self.image_files[index]),
            "label": self.labels[index],
        }


class CarlaImageMixDataset(data.Dataset):
    def __init__(
        self,
        carla_data_root: str,
    ):

        carla_image_files = []
        carla_labels = []

        carla_labels_dir = os.path.join(carla_data_root, 'labels')
        carla_images_dir = os.path.join(carla_data_root, 'images')
        for file in os.listdir(carla_labels_dir):
            if file.endswith('.json'):
                with open(os.path.join(carla_labels_dir, file), 'r') as f:
                    label_dict = json.load(f)
                    state = label_dict['state']
                    if not state:
                        continue
                    image_path = os.path.join(carla_images_dir, label_dict['image'])
                    if os.path.exists(image_path):
                        carla_image_files.append(image_path)
                        vehicle_transform = convert_dict_transform(label_dict['vehicle'])
                        camera_transform = convert_dict_transform(label_dict['camera'])
                        fov = label_dict['camera']['fov']
                        name = label_dict['name']
                        carla_labels.append([vehicle_transform, camera_transform, fov, name])
                    else:
                        raise FileNotFoundError(image_path)
        self.carla_image_files = carla_image_files
        self.carla_labels = carla_labels

    def __len__(self):
        return len(self.carla_image_files)

    def __getitem__(self, index):
        return {
            "image": cv2.imread(self.carla_image_files[index]),
            "label": self.carla_labels[index],
        }
