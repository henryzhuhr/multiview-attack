from typing import List

import os
import sys
import shutil
import time
import json

import cv2
import numpy as np

import carla

sys.path.append(os.getcwd())
from modules import types


class Args:

    data_root = "tmp/fromDAT"
    DAS_data_root = os.path.expandvars('$HOME/datasets/DAS-data/phy_attack/train')

    class Dirs:
        image = 'images'
        label = 'labels'
        segmentation = 'segmentations'

        @staticmethod
        def update():
            Args.Dirs.image = os.path.join(Args.data_root, Args.Dirs.image)
            Args.Dirs.label = os.path.join(Args.data_root, Args.Dirs.label)
            Args.Dirs.segmentation = os.path.join(Args.data_root, Args.Dirs.segmentation)


def main():
    os.makedirs(Args.Dirs.label, exist_ok=True)
    os.makedirs(Args.Dirs.image, exist_ok=True)

    for file in os.listdir(Args.DAS_data_root)[: 5]:
        if file.endswith('.npz'):
            file_name = os.path.splitext(file)[0]
            # if file_name !='data11216':
            #     continue
            data_path = os.path.join(Args.DAS_data_root, file)
            data = np.load(data_path)
            img: np.ndarray = data['img']
            veh_trans: np.ndarray = data['veh_trans'].astype(np.float64)
            cam_trans: np.ndarray = data['cam_trans'].astype(np.float64)

            print(file_name)
            print(veh_trans[0], veh_trans[1])
            print(cam_trans[0], cam_trans[1])

            cv2.imwrite(os.path.join(Args.Dirs.image, f'{file_name}.png'), img)
            with open(os.path.join(Args.Dirs.label, f'{file_name}.json'), 'w') as f:
                json.dump(
                    {
                        "name": file_name,
                        "image": f"{file_name}.png",
                        "state": True,
                        "map": "Town10HD",
                        "vehicle":
                            {
                                "location": {
                                    "x": veh_trans[0][0],
                                    "y": veh_trans[0][1],
                                    "z": veh_trans[0][2]
                                },
                                "rotation": {
                                    "pitch": veh_trans[1][0],
                                    "yaw": veh_trans[1][1],
                                    "roll": veh_trans[1][2]
                                }
                            },
                        "camera":
                            {
                                "location": {
                                    "x": cam_trans[0][0],
                                    "y": cam_trans[0][1],
                                    "z": cam_trans[0][2]# * 1.2
                                },
                                "rotation":
                                    {
                                        "pitch": cam_trans[1][0],
                                        "yaw": cam_trans[1][1],#,-90.9,
                                        "roll": cam_trans[1][2]
                                    },
                                "fov": 90
                            }
                    },
                    f,
                    indent=4,
                    ensure_ascii=False
                )


if __name__ == '__main__':
    Args.Dirs.update()
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('[Exit].')
