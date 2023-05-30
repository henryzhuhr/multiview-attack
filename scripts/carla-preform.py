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
import tqdm

sys.path.append(os.getcwd())
from models.data import types

tm_port = 8000

save_scene = False # 是否保存场景图


class Args:
    world_map = "Town01"
    data_root = f"data/train"

    class Dirs:
        image = 'images'
        scene = 'scenes'
        label = 'labels'

        @staticmethod
        def update():
            Args.Dirs.label = os.path.join(Args.data_root, Args.world_map, Args.Dirs.label)
            Args.Dirs.image = os.path.join(Args.data_root, Args.world_map, Args.Dirs.image)
            Args.Dirs.scene = os.path.join(Args.data_root, Args.world_map, Args.Dirs.scene)
            os.makedirs(Args.Dirs.image, exist_ok=True)
            # os.makedirs(Args.Dirs.scene, exist_ok=True)


Args.Dirs.update()


def main():
    label_dicts = get_labels(Args.Dirs.label)
    if len(label_dicts)==0:
        print("No label files")
        exit(0)

    actor_list: List[types.carla.Actor] = []
    vehicle_bps = []
    walkers_list = []
    camera_actor: types.carla.Actor = None
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.load_world(Args.world_map)
        # world = client.get_world()

        traffic_manager = client.get_trafficmanager(tm_port)      # 获取交通管理器
        traffic_manager.set_global_distance_to_leading_vehicle(2) # 与前车距离

        # 获取蓝图
        blueprint_library = world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter('vehicle')
        filter_out = [
                       # 'vehicle.tesla.model3',
        ]
        vehicle_bps = [x for x in vehicle_bps if (x.id not in filter_out)]
        vehicle_bps = vehicle_bps + vehicle_bps + vehicle_bps
        print(f" -- Get {len(vehicle_bps)} vehicles")

        # 获取出生点
        spawn_points = world.get_map().get_spawn_points()
        # 生成车辆
        pbar = tqdm.tqdm(vehicle_bps)
        for bp in pbar:
            shuffled_list = spawn_points.copy()
            random.shuffle(shuffled_list)
            for transform in shuffled_list:
                actor = world.try_spawn_actor(bp, transform)
                if actor is not None:
                    actor_list.append(actor)
                    actor.set_autopilot(True)
                    pbar.set_description('created %s' % actor.type_id)
                    break

        # 加载 Model3
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

        # 加载 Camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(Settings.image_size[0]))
        camera_bp.set_attribute('image_size_y', str(Settings.image_size[1]))
        camera_bp.set_attribute('sensor_tick', str(Settings.sensor_tick))
        camera_bp.set_attribute('shutter_speed', str(Settings.shutter_speed))

        pbar = tqdm.tqdm(label_dicts.items())
        time.sleep(3)
        for label_file, label_dict in pbar:
            save_file = label_dict['name'] + ".png"
            save_file_path = os.path.join(Args.Dirs.image, save_file)


            pbar.set_description(label_dict['name'])
            if os.path.exists(save_file_path):
                continue

            vehicle_transform = dict_to_carla_transform(label_dict['vehicle'])
            camera_transform = dict_to_carla_transform(label_dict['camera'])

            camera_bp.set_attribute('fov', str(label_dict['camera']["fov"]))

            try:
                vehicle_actor: types.carla.Actor = world.spawn_actor(vehicle_bp, vehicle_transform)
            except:
                continue

            time.sleep(1)
            camera_actor: types.carla.Actor = world.spawn_actor(
                camera_bp,
                camera_transform,
                attach_to=vehicle_actor,
                attachment_type=carla.AttachmentType.Rigid,
            )

            # if not save_scene:
            #     vehicle_actor.destroy()
            #     time.sleep(2)

            time.sleep(2)
            camera_actor.listen(lambda image: save_img(image, Settings.image_size, save_file_path))

            time.sleep(2)

            if camera_actor:
                camera_actor.stop()
                carla.command.DestroyActor(camera_actor)
            if vehicle_actor:
                carla.command.DestroyActor(vehicle_actor)

    except KeyboardInterrupt:
        print(' - Keyboard Interrupt')
    finally:
        print('destroying actors')
        if camera_actor:
            camera_actor.destroy()
        print('destroy actors: ', len(actor_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])


def get_labels(label_dir: str):
    label_list = {}
    for file in os.listdir(label_dir):
        if file.endswith('.json'):
            label_file = os.path.join(label_dir, file)
            with open(label_file, 'r') as f:
                label_dict = json.load(f)
                save_file = label_dict['name'] + ".png"
                image_save = os.path.join(Args.Dirs.image, save_file)
                if not os.path.exists(image_save):
                    label_list[label_file] = label_dict
    return label_list


def dict_to_carla_transform(transform_dict: dict) -> types.carla.Transform:
    return carla.Transform(
        location=carla.Location(
            x=transform_dict['location']['x'],         #
            y=transform_dict['location']['y'],
            z=transform_dict['location']['z']
        ),
        rotation=carla.Rotation(
            pitch=transform_dict['rotation']['pitch'], #
            yaw=transform_dict['rotation']['yaw'],
            roll=transform_dict['rotation']['roll']
        ),
    )


class Settings:
    bloom_intensity: float = 0.675    # 0.675   Intensity for the bloom post-process effect, 0.0 for disabling it.
    fov: float = 90.0                 # 90.0    Horizontal field of view in degrees.
    fstop: float = 2.8                # 1.4     Opening of the camera lens. Aperture is 1/fstop with typical lens going down to f/1.2 (larger opening). Larger numbers will reduce the Depth of Field effect.
    image_size = [800, 800]           # 600     Image height in pixels.
    iso: float = 100                  # 100     The camera sensor sensitivity.
    gamma: float = 2.2                # 2.2     Target gamma value of the camera.
    lens_flare_intensity: float = 0.1 # 0.1     Intensity for the lens flare post-process effect, 0.0 for disabling it.
    sensor_tick: float = 0.1          # 0.0     Simulation seconds between sensor captures (ticks).
    shutter_speed: float = 3000.0     # 200.0   The camera shutter speed in seconds (1.0/s).


def save_img(image, image_sizes: List[int], save_path: str):
    # print(image)
    img_raw_bytes = np.array(image.raw_data)
    img_channel_4 = img_raw_bytes.reshape((image_sizes[0], image_sizes[1], 4))
    img_channel3 = img_channel_4[:, :, : 3]
    cv2.imwrite(save_path, img_channel3)


class Color:
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


if __name__ == '__main__':
    main()
