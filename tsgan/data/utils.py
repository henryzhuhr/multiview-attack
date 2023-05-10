import os
import json
from typing import Dict, List

from . import types

class CarlaDatasetDir:
    def __init__(self, root, data_type:str=None) -> None:
        base_root= os.path.join(root, data_type)  if data_type is not None else root
        self.images_dir = os.path.join(base_root, "scenes")
        self.labels_dir = os.path.join(base_root, "labels")
        self.segmentations_dir = os.path.join(base_root, "segmentations")

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

def load_carla_label(label_path):
    with open(label_path, "r") as f:
        label_dict = json.load(f)
    vehicle_transform = convert_dict_transform(label_dict["vehicle"])
    camera_transform = convert_dict_transform(label_dict["camera"])
    fov = label_dict["camera"]["fov"]
    name = label_dict["name"]
    return {
        "name": name,
        "vehicle_transform": vehicle_transform,
        "camera_transform": camera_transform,
        "fov": fov,
    }
    return vehicle_transform, camera_transform, fov, name