import json
import os, sys
from typing import Dict
import cv2
from torch.utils import data
from PIL import Image

sys.path.append(os.getcwd())


class Location:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        self.x = x
        self.y = y
        self.z = z


class Rotation:
    def __init__(self, pitch: float = 0., yaw: float = 0., roll: float = 0.) -> None:
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class Transform:
    def __init__(self, location=Location(0, 0, 0), rotation=Rotation(0, 0, 0)) -> None:
        self.location = location
        self.rotation = rotation


def convert_dict_transform(transform_dict: Dict):
    return Transform(
        location=Location(
            x=transform_dict['location']['x'], y=transform_dict['location']['y'], z=transform_dict['location']['z']
        ),
        rotation=Rotation(
            pitch=transform_dict['rotation']['pitch'],
            yaw=transform_dict['rotation']['yaw'],
            roll=transform_dict['rotation']['roll']
        )
    )


class DatasetDir:
    images = 'images'
    labels = 'labels'
    segmentations = 'segmentations'


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
        return (
            cv2.imread(self.image_files[index]),
            self.labels[index],
        )
