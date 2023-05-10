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
from tsgan import types


class Settings:

    data_root = "tmp/data"
    world_map = "Town10HD"

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

        [5, 4, 2, 90],
        [5, 5, 2.5, 90],
        [6, 4, 3, 90], 
        [7, 4, 2, 90],
    ]

    camera_distance = [5, 0, 2, 90]


def main():
    world_map = Settings.world_map
    start_time = time.time()
    client = None
    actor_list = []

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10)
        world = client.load_world(world_map)
        os.makedirs("tmp", exist_ok=True)
        print(f'Load Map:{CStr.blue}{world_map}{CStr.reset}', 'in %.2f s' % (time.time() - start_time))

        # 获取蓝图库 Get blueprint_library
        blueprint_library = world.get_blueprint_library()

        # 获取特定汽车蓝图 Get vehicle blueprint
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

        # 获取全部出生点 Get all recommended spawn points
        spawn_points: List[types.carla.Transform] = world.get_map().get_spawn_points()

        # 随机选择一个出生点
        spawn_point = random.choice(spawn_points)
        spawn_point = spawn_points[0]
        vehicle_transform: types.carla.Transform = spawn_point
        vehicle_actor: types.carla.Actor = world.spawn_actor(vehicle_bp, vehicle_transform)
        # actor_list.append(vehicle_actor)

        (x, y, z, fov) = Settings.camera_distance

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(800))
        camera_bp.set_attribute('image_size_y', str(800))
        camera_bp.set_attribute('sensor_tick', str(0.0))
        camera_bp.set_attribute('shutter_speed', str(1500))
        camera_bp.set_attribute('fov', str(fov))

        pitch, yaw, roll = get_rotation_by_center_actor(x, y, z)
        camera_transform: types.carla.Transform = carla.Transform(
            location=carla.Location(x, y, z), rotation=carla.Rotation(pitch, yaw, roll)
        )

        rgb_camera_actor: types.carla.Actor = world.spawn_actor(
            camera_bp, camera_transform, attach_to=vehicle_actor, attachment_type=carla.AttachmentType.Rigid
        )
        
        if False:
            time.sleep(1)
            camera_abs_transform = rgb_camera_actor.get_transform()
            rgb_camera_actor: types.carla.Actor = world.spawn_actor(camera_bp, camera_abs_transform)
            vehicle_actor.destroy()
            time.sleep(1)
        actor_list.append(rgb_camera_actor)

        rgb_camera_actor.listen(lambda image: save_img(image, [800, 800], "tmp/rgb.png"))

        time.sleep(1.2)
        rgb_camera_actor.stop()

    except RuntimeError as e:
        print(CStr.red, '[RuntimeError]', CStr.reset, f'in Map:{world_map}', e)
    finally:
        if client:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])


def save_img(image, image_sizes: List[int], save_path: str, bounding_boxes=None):
    # print(image)
    img_raw_bytes = np.array(image.raw_data)
    img_channel_4 = img_raw_bytes.reshape((image_sizes[0], image_sizes[1], 4))
    img_channel3 = img_channel_4[:, :, : 3]
    # img_channel3 = ClientSideBoundingBoxes.draw_bounding_boxes(img_channel3, bounding_boxes)
    if True:
        cv2.imwrite(save_path, img_channel3)


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


class CStr:
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
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('[Exit].')
