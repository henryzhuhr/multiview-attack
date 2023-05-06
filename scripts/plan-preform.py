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
from tsgan import types


class Args:
    data_root = "tmp/data"

    class Dirs:
        image = 'images-src'
        label = 'labels'
        segmentation = 'segmentations'

        @staticmethod
        def update():
            Args.Dirs.image = os.path.join(Args.data_root, Args.Dirs.image)
            Args.Dirs.label = os.path.join(Args.data_root, Args.Dirs.label)
            Args.Dirs.segmentation = os.path.join(Args.data_root, Args.Dirs.segmentation)


def retrieve_from_label(label_file: str):
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

    with open(label_file, 'r') as f:
        label_dict = json.load(f)
        if not check_dict_type(label_dict, types.label.valid_label_type_dict):
            return False
        # if label_dict['state']:
        #     return True
    plan_name = label_dict['name']
    world_map = label_dict['map']
    

    print(plan_name)

    

    start_time = time.time()
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10)
        world = client.load_world(world_map)

        vehicle_transform = dict_to_carla_transform(label_dict['vehicle'])
        camera_transform = dict_to_carla_transform(label_dict['camera'])


        # Get blueprint_library
        blueprint_library = world.get_blueprint_library()

        # Get vehicle blueprint
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle_actor: types.carla.Actor = world.spawn_actor(vehicle_bp, vehicle_transform)
        actor_list.append(vehicle_actor)

        CameraManager.Settings.fov=label_dict['camera']['fov']
        camera_manager=CameraManager(blueprint_library)

        rgb_camera_actor: types.carla.Actor = world.spawn_actor(
            camera_manager.get_camera_rgb_blueprint(),
            camera_transform,
            attach_to=vehicle_actor,
            attachment_type=carla.AttachmentType.Rigid
        )
        actor_list.append(rgb_camera_actor)

        seg_camera_actor = world.spawn_actor(
            camera_manager.get_camera_instance_segmentation_blueprint(),
            camera_transform,
            attach_to=vehicle_actor,
            attachment_type=carla.AttachmentType.Rigid
        )
        actor_list.append(seg_camera_actor)

        save_file = label_dict['image']
        image_save = os.path.join(Args.Dirs.image, save_file)
        iseg_save = os.path.join(Args.Dirs.segmentation, save_file)

        rgb_camera_actor.listen(
            lambda image: CameraManager.save_img(
                image,                                                                      #
                [CameraManager.Settings.image_size_x, CameraManager.Settings.image_size_y],
                image_save
            )
        )
        seg_camera_actor.listen(
            lambda image: CameraManager.save_img(
                image,                                                                      #
                [CameraManager.Settings.image_size_x, CameraManager.Settings.image_size_y],
                iseg_save
            )
        )
        time.sleep(1.2)

        

        # relative_location: types.carla.Location = carla.Location(
        #     x=vehicle_transform.location.x-rgb_camera_actor.get_transform().location.x,              #
        #     y=vehicle_transform.location.y-rgb_camera_actor.get_transform().location.y,
        #     z=vehicle_transform.location.z-rgb_camera_actor.get_transform().location.z
        # )
        # relative_rotation: types.carla.Rotation = carla.Rotation(
        #     pitch=vehicle_transform.rotation.pitch,              #
        #     yaw=vehicle_transform.rotation.yaw,
        #     roll=vehicle_transform.rotation.roll,
        # )
        # empt_camera_transform: types.carla.Transform=carla.Transform(location=relative_location, rotation=relative_rotation)
        # print()
        # print()
        # print('[vehicle transform]   ', vehicle_actor.get_transform())        
        # print('[camera transform] abs', rgb_camera_actor.get_transform())
        # print('[camera transform] div', empt_camera_transform)
        # print('[camera transform] rel', camera_transform)
        

        state = True
        print(f'{ColorConsole.greenl}[State]{ColorConsole.reset}', plan_name, end=': ')
        for file in [image_save, iseg_save]:
            if not os.path.exists(file):
                state = False
                print(f'{file}', end=', ')
        print(' NOT save' if not state else 'Finish', 'in %.2f s' % (time.time() - start_time))
        rgb_camera_actor.stop()
        seg_camera_actor.stop()

        with open(label_file, 'w') as f:
            label_dict['state'] = state # 改变状态
            json.dump(
                label_dict, f, indent=4, ensure_ascii=False
            )                                               # print(plan_name, save_file, vehicle_transform, camera_transform)
    except RuntimeError as e:
        print(ColorConsole.red, '[RuntimeError]', ColorConsole.reset, f'in Map:{world_map}', e)
    finally:
        if client:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])


def main():
    if not os.path.exists(Args.Dirs.image):
        os.makedirs(Args.Dirs.image, exist_ok=True)
    if not os.path.exists(Args.Dirs.segmentation):
        os.makedirs(Args.Dirs.segmentation, exist_ok=True)

    for file in os.listdir(Args.Dirs.label):
        if file.endswith('.json'):
            retrieve_from_label(os.path.join(Args.Dirs.label, file))
            # break


def check_dict_type(checked_dict: dict, valid_dict: dict):
    for key in valid_dict.keys():
        if isinstance(valid_dict[key], dict):
            if not check_dict_type(checked_dict[key], valid_dict[key]):
                return False
        else:
            if not isinstance(checked_dict[key], valid_dict[key]):
                print(f'[Invalid] {key} got {type(checked_dict[key])}, but expected {valid_dict[key]}')
                return False
    return True


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


class CameraManager:
    class Settings:
        bloom_intensity: float = 0.675    # 0.675   Intensity for the bloom post-process effect, 0.0 for disabling it.
        fov: float = 90.0                 # 90.0    Horizontal field of view in degrees.
        fstop: float = 1.4                # 1.4     Opening of the camera lens. Aperture is 1/fstop with typical lens going down to f/1.2 (larger opening). Larger numbers will reduce the Depth of Field effect.
        image_size_x: int = 800           # 800     Image width in pixels.
        image_size_y: int = 800           # 600     Image height in pixels.
        iso: float = 100                  # 100     The camera sensor sensitivity.
        gamma: float = 2.2                # 2.2     Target gamma value of the camera.
        lens_flare_intensity: float = 0.1 # 0.1     Intensity for the lens flare post-process effect, 0.0 for disabling it.
        sensor_tick: float = 0.0          # 0.0     Simulation seconds between sensor captures (ticks).
        shutter_speed: float = 1500.0     # 200.0   The camera shutter speed in seconds (1.0/s).

    def __init__(self, blueprint_library) -> None:
        self.blueprint_library = blueprint_library

    def get_camera_rgb_blueprint(self):
        blueprint_library = self.blueprint_library
        bp = blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.Settings.image_size_x))
        bp.set_attribute('image_size_y', str(self.Settings.image_size_y))
        bp.set_attribute('sensor_tick', str(self.Settings.sensor_tick))
        bp.set_attribute('shutter_speed', str(self.Settings.shutter_speed))
        print('fov',self.Settings.fov)
        bp.set_attribute('fov', str(self.Settings.fov))
        return bp

    def get_camera_instance_segmentation_blueprint(self):
        blueprint_library = self.blueprint_library
        bp = blueprint_library.find('sensor.camera.instance_segmentation')
        bp.set_attribute('image_size_x', str(self.Settings.image_size_x))
        bp.set_attribute('image_size_y', str(self.Settings.image_size_y))
        bp.set_attribute('sensor_tick', str(self.Settings.sensor_tick))
        bp.set_attribute('fov', str(self.Settings.fov))
        return bp

    @staticmethod
    def save_img(image, image_sizes: List[int], save_path: str, bounding_boxes=None):
        # print(image)
        img_raw_bytes = np.array(image.raw_data)
        img_channel_4 = img_raw_bytes.reshape((image_sizes[0], image_sizes[1], 4))
        img_channel3 = img_channel_4[:, :, : 3]
        # img_channel3 = ClientSideBoundingBoxes.draw_bounding_boxes(img_channel3, bounding_boxes)
        if True:
            cv2.imwrite(save_path, img_channel3)


if __name__ == '__main__':
    Args.Dirs.update()

    try:
        main()

    except KeyboardInterrupt:
        pass
    finally:
        print('[Exit].')
