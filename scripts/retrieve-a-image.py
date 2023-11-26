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


class Settings:

    data_root = "tmp/data"
    world_map = "Town10HD"
    save_dir = "data/train/Test"

    # camera_distances = [
    #     # [4, 2, 3, 90],
    #     # [5, 3, 3, 90],
    #     # [6, 4, 3, 90],
    #     # [5, 5, 3, 90],
    #     # [4, -2, 3, 90],
    #     # [5, -3, 3, 90],
    #     # [6, -4, 3, 90],
    #     # [5, -5, 3, 90],

    #     # *([4, i, 3, 90] for i in [3,4,5]),
    #     # *([4, -i, 3, 90] for i in [3,4,5]),
    #     # *([0, i, 1.5, 90] for i in [4,5,6,7]),
    #     # *([0, -i, 1.5, 90] for i in [4,5,6,7]),

    # ]
    # camera_distances = [
    #     [0, 4, 1.5, 90],
    #     [0, -4, 1.5, 90],
    #     [0.1, 0, 3.5, 90],
    #     [3, 0, 2, 90],
    #     [2, 0, 3, 90],
    # ]
    # [x,y,z,fov]
    # camera_distance = [-4, 0, 1, 90]
    camera_distance = [0, -6, 1.5, 90]

    camera_distances = []


get_save_name = lambda x, y, z, fov: f"Town10HD-{x}_{y}_{z}_{fov}"

dz_list = [
    [4, 3],
    [5, 4],
    [6, 5],
]

for d, z in dz_list:
    cds = [
        *[
            [0, d, z, 90],
            [0, -d, z, 90],
            [d, 0, z, 90],
            [-d, 0, z, 90],
        ],
        *[
            [d, d, z, 90],
            [d, -d, z, 90],
            [d, d, z, 90],
            [-d, d, z, 90],
        ],
        *[
            [d, d / 2, z, 90],
            [d, -d / 2, z, 90],
            [d, d / 2, z, 90],
            [-d, d / 2, z, 90],
        ],
        *[
            [d / 2, d, z, 90],
            [d / 2, -d, z, 90],
            [d / 2, d, z, 90],
            [-d / 2, d, z, 90],
        ],
        *[
            [d / 4, d, z, 90],
            [d / 4, -d, z, 90],
            [d / 4, d, z, 90],
            [-d / 4, d, z, 90],
        ],
        *[
            [d, d / 4, z, 90],
            [d, -d / 4, z, 90],
            [d, d / 4, z, 90],
            [-d, d / 4, z, 90],
        ],
    ]
    for cd in cds:
        Settings.camera_distances.append(cd)
"""
dir=data/train/Town
mkdir -p $dir/images
cp -r data/samples/*.png $dir/images
mkdir -p $dir/labels
cp -r data/samples/*.json $dir/labels
"""
print("distance: ", len(Settings.camera_distances))
# for cd in cds:
#     print(cd)

# exit()


def main():
    world_map = Settings.world_map
    start_time = time.time()
    client = None
    actor_list = []
    os.makedirs(images_save_dir := f'{Settings.save_dir}/images', exist_ok=True)
    os.makedirs(labels_save_dir := f'{Settings.save_dir}/labels', exist_ok=True)

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
        spawn_point = spawn_points[1]
        vehicle_transform: types.carla.Transform = spawn_point
        vehicle_actor: types.carla.Actor = world.spawn_actor(vehicle_bp, vehicle_transform)
        # actor_list.append(vehicle_actor)

        for (x, y, z, fov) in Settings.camera_distances:
            # (x, y, z, fov) = Settings.camera_distance

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

            save_name = get_save_name(x, y, z, fov)
            with open(f'{labels_save_dir}/{save_name}.json', 'w') as f_label:
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
            rgb_camera_actor.listen(lambda image: save_img(image, [800, 800], f"{images_save_dir}/{save_name}.png"))

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
