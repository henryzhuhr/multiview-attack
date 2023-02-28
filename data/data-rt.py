import os
import shutil
import time
import json
from typing import List

import cv2
import numpy as np

import carla


class ARGS:
    data_root = 'tmp/'
    is_test = False
    autopilot = False

    [
        'Town01',
        'Town04',
        'Town05',
        'Town06',
        'Town07',
        'Town08',
        'Town09',
    ]
    [
        [3, 2, 2, 100],
        [3, 2, 2, 00],
        [3, 2, 3, 100],
        [4, 2.5, 2, 90],
        [4, 2.5, 3, 90],
        [4, 2.5, 4, 90],
        [10, 6, 3, 80],
        [10, 6, 6, 80],
        [12, 8, 7, 70],
        [12, 8, 5, 70],
        [15, 10, 10, 60],
        [15, 10, 5, 60],
    ]
    world_map = 'Town01'
    camera_distance = [4, 2, 3, 90]

    class CameraSettings:
        bloom_intensity: float = 0.675    # 0.675   Intensity for the bloom post-process effect, 0.0 for disabling it.
        fov: float = 90.0                 # 90.0    Horizontal field of view in degrees.
        fstop: float = 1.4                # 1.4     Opening of the camera lens. Aperture is 1/fstop with typical lens going down to f/1.2 (larger opening). Larger numbers will reduce the Depth of Field effect.
        image_size_x: int = 800           # 800     Image width in pixels.
        image_size_y: int = 800           # 600     Image height in pixels.
        iso: float = 100                  # 100     The camera sensor sensitivity.
        gamma: float = 2.2                # 2.2     Target gamma value of the camera.
        lens_flare_intensity: float = 0.1 # 0.1     Intensity for the lens flare post-process effect, 0.0 for disabling it.
        sensor_tick: float = 0.1          # 0.0     Simulation seconds between sensor captures (ticks).
        shutter_speed: float = 1200.0     # 200.0   The camera shutter speed in seconds (1.0/s).


def main():
    current_time = time.time()
    client = None
    actor_list = []
    world_map = ARGS.world_map

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(3)
        world = client.load_world(world_map)
        print(' Load World(%s) in %.2f s' % (world_map, time.time() - current_time))
        current_time = time.time()

        blueprint_library = world.get_blueprint_library()

        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]

        model3_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle_actor = world.spawn_actor(model3_bp, spawn_point)
        actor_list.append(vehicle_actor)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(ARGS.CameraSettings.image_size_x))
        camera_bp.set_attribute('image_size_y', str(ARGS.CameraSettings.image_size_y))
        camera_bp.set_attribute('sensor_tick', str(ARGS.CameraSettings.sensor_tick))
        camera_bp.set_attribute('shutter_speed', str(ARGS.CameraSettings.shutter_speed))
        x, y, z, fov = ARGS.camera_distance
        camera_bp.set_attribute('fov', str(fov))
        pitch, yaw, roll = Utils.get_rotation_by_center_actor(x, y, z)
        camera_transform = carla.Transform(
            location=carla.Location(x=x, y=y, z=z), rotation=carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
        )
        
        camera_actor = world.spawn_actor(
            camera_bp, camera_transform, attach_to=vehicle_actor, attachment_type=carla.AttachmentType.Rigid
        )
        
        actor_list.append(camera_actor)
        camera_actor.listen(
            lambda image: Utils.save_img(
                image,                                                                #
                [ARGS.CameraSettings.image_size_x, ARGS.CameraSettings.image_size_y],
                os.path.join(ARGS.data_root, 'tmp.png')
            )
        )
        time.sleep(0.5)
        print('camera_transform',camera_transform)
        print('vehicle_transform',vehicle_actor.get_transform())
        print('camera_transform',camera_actor.get_transform())
        camera_actor.stop()

    except RuntimeError as e:
        print(e)
    finally:
        print(' Finish in %.2f s' % (time.time() - current_time))
        print('... Destroying All Actors')
        if client:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])


class Utils:
    @staticmethod
    def extend_xyz(x, y, z):
        return [
                       # [x, 0, z],
                       # [x, y, z],
                       # [x, 0, z],
            [x, y, z],
                       # [0, y, z],
                       # [-x, y, z],
                       # [-x, 0, z],
                       # [-x, -y, z],
                       # [0, -y, z],
                       # [x, -y, z],
        ]

    @staticmethod
    def get_rotation_by_center_actor(x=0., y=0., z=0.):
        """
        Get Rotation Params by setting the actor in the center of view
        """
        x = x + 1e-9
        y = y + 1e-9
        # Step 1:  Since x can be negtiva, you do not change the symbol
        yaw = (180 if x > 0 else 0) + np.arctan(y / x) * (180 / np.pi)
        # Step 2:  pith is always negitive to see down
        pitch = -np.arctan(z / (np.sqrt(x * x + y * y))) * (180 / np.pi)
        return (pitch, yaw, roll := 0.)

    @staticmethod
    def save_img(image, image_sizes: List[int], save_path, bounding_boxes=None):
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


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('[Exit].')
