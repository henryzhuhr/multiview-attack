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
from models.data import types

is_eval = True     # 是否保存场景图


class Settings:
    world_map = "Town10HD"
    data_root = "data/"
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
    train_camera_distances = [ # train
        [5, 2, 90],
        [6, 3, 90],
        [7, 4, 90],
        [8, 5, 90],
    ]
    eval_camera_distances = [  # eval
        [5, 2, 90],
        [8, 3, 90],
        [12, 4, 90],
        [16, 5, 90],
    ]
    camera_distances = train_camera_distances


def generate_plan(world_map: str, data_root: str):
    start_time = time.time()
    client = None
    actor_list = []

    label_save_dir = os.path.join(data_root, 'labels')
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
        data_list = []
        for spawn_point_index, spawn_point in enumerate(spawn_points):
            vehicle_transform: types.carla.Transform = spawn_point

            # 遍历相机距离
            for distance_index, (d, z, fov) in enumerate(Settings.camera_distances):
                # 遍历相机方位
                for direction_index, (x, y) in enumerate(Utils.expand_coordinates(d)):
                    save_name = "%s-point_%04d-distance_%03d-direction_%d" % (
                        world_map, spawn_point_index, distance_index, direction_index
                    )

                    pitch, yaw, roll = Utils.get_rotation_by_center_actor(x, y, z)

                    camera_transform: types.carla.Transform = carla.Transform(
                        location=carla.Location(x, y, z), rotation=carla.Rotation(pitch, yaw, roll)
                    )
                    data_list.append([save_name, world_map, vehicle_transform, camera_transform, fov])

        random.shuffle(data_list)
        # data_list = random.sample(data_list, 500)
        for [save_name, world_map, vehicle_transform, camera_transform, fov] in data_list:
            with open(os.path.join(label_save_dir, f'{save_name}.json'), 'w') as f_label:
                json.dump(
                    {
                        "name": save_name,
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
        data_root = f"{Settings.data_root}/{world_map}"
        generate_plan(world_map, data_root)


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
    def expand_coordinates(d):
        import math
        points = [
            [d, 0], [-d, 0], [0, d], [0, -d], [d * math.cos(math.radians(45)), d * math.sin(math.radians(45))],
            [d * math.cos(math.radians(135)), d * math.sin(math.radians(135))],
            [d * math.cos(math.radians(225)), d * math.sin(math.radians(225))],
            [d * math.cos(math.radians(315)), d * math.sin(math.radians(315))]
        ]
        return points

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
