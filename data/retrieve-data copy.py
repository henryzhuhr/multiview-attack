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
    autopilot = False
    maps = ['Town01', 'Town04', 'Town05', 'Town06', 'Town07']

    # [x,y,z,fov]
    camera_distances = [
                         # [3, 2, 2, 100],
                         # [3, 2, 2, 00],
                         # [3, 2, 3, 100],
                         # [4, 2.5, 2, 90],
        [4, 2.5, 3, 90],
                         # [4, 2.5, 4, 90],
                         # [10, 6, 3, 80],
        [10, 6, 6, 80],
                         # [12, 8, 7, 70],
                         # [12, 8, 5, 70],
                         # [15, 10, 10, 60],
                         # [15, 10, 5, 60],
    ]


def retrieve_from_map(world_map: str):
    start_time = time.time()
    client = None
    actor_list = []

    pilot_type = 'auto' if Settings.autopilot else 'static'

    save_dir_map = {
        name: os.path.join(Settings.data_root, name, world_map, pilot_type)
        for name in [
            'images',
            'segmentations',
            'labels',
        ]
    }

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(60)
        world = client.load_world(world_map)
        print(f'Load Map:{ColorConsole.blue}{world_map}{ColorConsole.reset}', 'in %.2f s' % (time.time() - start_time))

        for d in save_dir_map.values():
            # if os.path.exists(d):
            #     shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        # Get blueprint_library
        blueprint_library = world.get_blueprint_library()

        # Get vehicle blueprint
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

        # Get all recommended spawn points
        spawn_points: List[types.carla.Transform] = world.get_map().get_spawn_points()[: 10]

        camera_manager = CameraManager(blueprint_library)
        rgb_camera_bp = camera_manager.get_camera_rgb_blueprint()
        seg_camera_bp = camera_manager.get_camera_instance_segmentation_blueprint()

        vehicle_actor: types.carla.Actor = world.spawn_actor(vehicle_bp, spawn_points[0])
        if Settings.autopilot:
            vehicle_actor.set_autopilot(True)

        for spawn_point_index, spawn_point in enumerate(spawn_points):
            vehicle_actor.set_transform(spawn_point)

            for distance_index, (x, y, z, fov) in enumerate(Settings.camera_distances):
                for direction_index, (x, y, z) in enumerate(Utils.extend_xyz(x, y, z)):

                    save_name = "%s-sp_%04d-dis_%03d-dir_%d" % (
                        pilot_type, spawn_point_index, distance_index, direction_index
                    )

                    pitch, yaw, roll = Utils.get_rotation_by_center_actor(x, y, z)

                    camera_transform: types.carla.Transform = carla.Transform(
                        location=carla.Location(x=x, y=y, z=z),
                        rotation=carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
                    )
                    # spawn rgb camera
                    rgb_camera_actor = world.spawn_actor(
                        rgb_camera_bp,
                        camera_transform,
                        attach_to=vehicle_actor,
                        attachment_type=carla.AttachmentType.Rigid
                    )
                    seg_camera_actor = world.spawn_actor(
                        seg_camera_bp,
                        camera_transform,
                        attach_to=vehicle_actor,
                        attachment_type=carla.AttachmentType.Rigid
                    )
                    time.sleep(1)

                    rgb_camera_actor.listen(
                        lambda image: CameraManager.save_img(
                            image,                                                                      #
                            [CameraManager.Settings.image_size_x, CameraManager.Settings.image_size_y],
                            os.path.join(save_dir_map['images'], f'{save_name}.png')
                        )
                    )
                    seg_camera_actor.listen(
                        lambda image: CameraManager.save_img(
                            image,                                                                      #
                            [CameraManager.Settings.image_size_x, CameraManager.Settings.image_size_y],
                            os.path.join(save_dir_map['segmentations'], f'{save_name}.png')
                        )
                    )

                    # actor_list.append(rgb_camera_actor)
                    # actor_list.append(seg_camera_actor)

                    vehicle_t = vehicle_actor.get_transform()
                    with open(os.path.join(save_dir_map['labels'], f'{save_name}.json'), 'w') as f_label:
                        json.dump(
                            {
                                "image": save_name + ".png",
                                "state": 0,
                                "map": world_map,
                                "pilot_type": pilot_type,
                                "vehicle":
                                    {
                                        "location":
                                            {
                                                "x": vehicle_t.location.x,
                                                "y": vehicle_t.location.y,
                                                "z": vehicle_t.location.z,
                                            },
                                        "rotation":
                                            {
                                                "pitch": vehicle_t.rotation.pitch,
                                                "yaw": vehicle_t.rotation.yaw,
                                                "roll": vehicle_t.rotation.roll,
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
                                    }
                            },
                            f_label,
                            indent=4,
                            ensure_ascii=False
                        )

                    print('\033[1;32m[Map]\033[0m', world_map, end=' ')
                    print('\033[1;32m[Save]\033[0m', end=' ')
                    print(save_name, end=' ')
                    print('\033[1;32m[Point]\033[0m', end=' ')
                    print(spawn_point_index, end=' ')
                    print('\033[1;32m[Direction]\033[0m', end=' ')
                    print((x, y, z, fov), end=' ')
                    print()
                    time.sleep(1)

                    rgb_camera_actor.stop()
                    carla.command.DestroyActor(rgb_camera_actor)

                    seg_camera_actor.stop()
                    carla.command.DestroyActor(seg_camera_actor)

            # carla.command.DestroyActor(vehicle_actor)
            # client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            # time.sleep(1)
        print()

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
    time.sleep(2)


def main():
    for world_map in Settings.maps:
        retrieve_from_map(world_map)


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

    @staticmethod
    def save_seg_img(image, image_sizes: List[int], save_path, bounding_boxes=None):
        # print(image)
        img_raw_bytes = np.array(image.raw_data)
        img_channel_4 = img_raw_bytes.reshape((image_sizes[0], image_sizes[1], 4))
        img_channel3 = img_channel_4[:, :, : 3]
        # img_channel3 = ClientSideBoundingBoxes.draw_bounding_boxes(img_channel3, bounding_boxes)
        if True:
            cv2.imwrite(save_path, img_channel3)


class Utils:
    @staticmethod
    def extend_xyz(x, y, z):
        return [
            [x, 0, z], #
            [x, y, z],
                       # [x, 0, z],
                       # [x, y, z],
                       # [0, y, z],
                       # [-x, y, z],
                       # [-x, 0, z],
                       # [-x, -y, z],
                       # [0, -y, z],
                       # [x, -y, z]
        ]

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
