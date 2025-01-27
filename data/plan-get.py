import random
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


class Settings:

    data_root = "tmp/data"
    # client.get_available_maps()
    maps = [
        # 'Town01',
        # 'Town02',
        # 'Town03',
        # 'Town04',
        # 'Town06',
        # 'Town07',
        # 'Town05',
        'Town10HD',
                    # 'Town11',
    ]

    # [x,y,z,fov]
    camera_distances = [
                         # [3, 2, 2, 100],
                         # [3, 2, 2, 00],
        # [3, 2, 3, 90],
                         # [4, 2.5, 2, 90],
                         # [4, 2.5, 3, 90],
        [4, 2.5, 4, 90],
        # [10, 6, 3, 90],
        [8, 6, 3, 90],
                         # [12, 8, 7, 70],
        [12, 8, 5, 90],
                         # [15, 10, 10, 60],
        # [15, 10, 5, 90],
    ]


def generate_plan(world_map: str):
    start_time = time.time()
    client = None
    actor_list = []

    label_save_dir = os.path.join(Settings.data_root, 'labels')
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10)
        world = client.load_world(world_map)
        os.makedirs(label_save_dir, exist_ok=True)
        print(f'Load Map:{ColorConsole.blue}{world_map}{ColorConsole.reset}', 'in %.2f s' % (time.time() - start_time))

        # 获取蓝图库 Get blueprint_library
        blueprint_library = world.get_blueprint_library()

        # 获取特定汽车蓝图 Get vehicle blueprint
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

        # 获取全部出生点 Get all recommended spawn points
        spawn_points: List[types.carla.Transform] = world.get_map().get_spawn_points()

        # 遍历全部出生点
        for spawn_point_index, spawn_point in enumerate(spawn_points):
            vehicle_transform: types.carla.Transform = spawn_point

            # 遍历相机距离
            for distance_index, (x, y, z, fov) in enumerate(Settings.camera_distances):
                # 遍历相机方位
                for direction_index, (x, y, z) in enumerate(Utils.extend_xyz(x, y, z)):
                    save_name = "%s-point_%04d-distance_%03d-direction_%d" % (
                        world_map, spawn_point_index, distance_index, direction_index
                    )

                    pitch, yaw, roll = Utils.get_rotation_by_center_actor(x, y, z)

                    camera_transform: types.carla.Transform = carla.Transform(
                        location=carla.Location(x, y, z), rotation=carla.Rotation(pitch, yaw, roll)
                    )

                    with open(os.path.join(label_save_dir, f'{save_name}.json'), 'w') as f_label:
                        json.dump(
                            {
                                "name": save_name,
                                "image": save_name + ".png",
                                "state": False,
                                "map": world_map,
                                "vehicle":
                                    {
                                        "location":
                                            {
                                                "x": vehicle_transform.location.x,
                                                "y": vehicle_transform.location.y,
                                                "z": vehicle_transform.location.z,
                                            },
                                        "rotation":
                                            {
                                                "pitch": vehicle_transform.rotation.pitch,
                                                "yaw": vehicle_transform.rotation.yaw,
                                                "roll": vehicle_transform.rotation.roll,
                                            }
                                    },
                                "camera":
                                    {
                                        "location":
                                            {
                                                "x": camera_transform.location.x,
                                                "y": camera_transform.location.y,
                                                "z": camera_transform.location.z,
                                            },
                                        "rotation":
                                            {
                                                "pitch": camera_transform.rotation.pitch,
                                                "yaw": camera_transform.rotation.yaw,
                                                "roll": camera_transform.rotation.roll,
                                            },
                                        "fov": fov
                                    },
                            },
                            f_label,
                            indent=4,
                            ensure_ascii=False
                        )
                    # print(
                    #     f'\033[1;32m[Map]\033[0m {world_map}',  #
                    #     f'\033[1;32m[Save]\033[0m {save_name}', #
                    # )

    except RuntimeError as e:
        print(ColorConsole.red, '[RuntimeError]', ColorConsole.reset, f'in Map:{world_map}', e)
    else:
        print(
            f'Finish Map:{ColorConsole.blue}{world_map}{ColorConsole.reset}', 'in %.2f s' % (time.time() - start_time)
        )
    finally:
        if client:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print()


def main():
    for world_map in Settings.maps:
        generate_plan(world_map)


class ColorConsole:
    reset = '\033[0m'
    red = '\033[00;31m'
    redl = '\033[01;31m'
    green = '\033[00;32m'
    greenl = '\033[01;32m'
    yellow = '\033[00;33m'
    yellowl = '\033[01;33m'
    blue = '\033[00;34m'
    bluel = '\033[01;34m'
    magenta = '\033[00;35m'
    magental = '\033[01;35m'
    purple = '\033[00;35m'
    purplel = '\033[01;35m'
    cyan = '\033[00;36m'
    cyanl = '\033[01;36m'
    white = '\033[01;37m'
    grayl = '\033[00;37m'


class Utils:
    @staticmethod
    def extend_xyz(x, y, z):
        e=[
            [x, 0, z],
            [x, y, z],
            [x, 0, z],
            [x, y, z],
            [0, y, z],
            [-x, y, z],
            [-x, 0, z],
            [-x, -y, z],
            [0, -y, z],
            [x, -y, z],
        ]
        return random.sample(e, 3)

    @staticmethod
    def get_rotation_by_center_actor(x=0., y=0., z=0.):
        """
        Get Rotation Params by putting the actor in the center of view
        """
        x = x + 1e-9
        y = y + 1e-9
        # Step 1:  Since x can be negtiva, you do not change the symbol
        yaw = (180 if x > 0 else 0) + np.arctan(y / x) * (180 / np.pi)
        # Step 2:  pith is always negitive to see down
        pitch = -np.arctan(z / (np.sqrt(x * x + y * y))) * (180 / np.pi)
        return (pitch, yaw, roll := 0.)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('[Exit].')
