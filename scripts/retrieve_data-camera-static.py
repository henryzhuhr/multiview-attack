#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
from math import gamma
import os
import shutil
import time
import json
from typing import List

import cv2
import numpy as np

import carla

# def get_args():
#     argparser = argparse.ArgumentParser(description=__doc__)
#     args = argparser.parse_args()
#     return args

data_save_dir = 'tmp/data'
is_test = False
camera_distance_list = [
                         # [x,y,z,fov]
                         # [3, 2, 2, 100],
                         # [3, 2, 2, 100],
                         # [3, 2, 3, 100],
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
MAPS = [
    'Town01',
    'Town02',
    'Town03',
    'Town04',
    'Town05',
    'Town06',
    'Town07',
    'Town08',
    'Town09',
]
is_autopilot = False


class CameraSettings:
    bloom_intensity: float = 0.675    # 0.675   Intensity for the bloom post-process effect, 0.0 for disabling it.
    fov: float = 90.0                 # 90.0    Horizontal field of view in degrees.
    fstop: float = 1.4                # 1.4     Opening of the camera lens. Aperture is 1/fstop with typical lens going down to f/1.2 (larger opening). Larger numbers will reduce the Depth of Field effect.
    image_size_x: int = 800           # 800     Image width in pixels.
    image_size_y: int = 800           # 600     Image height in pixels.
    iso: float = 100                  # 100     The camera sensor sensitivity.
    gamma: float = 2.2                # 2.2     Target gamma value of the camera.
    lens_flare_intensity: float = 0.1 # 0.1     Intensity for the lens flare post-process effect, 0.0 for disabling it.
    sensor_tick: float = 0.0          # 0.0     Simulation seconds between sensor captures (ticks).
    shutter_speed: float = 1200.0     # 200.0   The camera shutter speed in seconds (1.0/s).


def main(world_map: str):
    actor_list = []

    for dir in [
        os.path.join(data_save_dir, name, world_map) for name in [
            'images',
            'segmentations',
            'labels',
        ]
    ]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)

    current_time = time.time()
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10)
        world = client.load_world(world_map)
        print(' Load World in %.2f s' % (time.time() - current_time))
        current_time = time.time()

        blueprint_library = world.get_blueprint_library()
        with open(os.path.join(data_save_dir, 'blueprint_library.json'), 'w') as f:
            bp_list = []
            for index, bp in enumerate(blueprint_library):
                bp_list.append({"index": index, "id": bp.id, "tags": bp.tags})
            json.dump({"blueprint_library": bp_list}, f, indent=4, ensure_ascii=False)

        # blueprint model3 vehicle
        model3_bp = blueprint_library.find('vehicle.tesla.model3')

        # blueprint camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CameraSettings.image_size_x))
        camera_bp.set_attribute('image_size_y', str(CameraSettings.image_size_y))
        camera_bp.set_attribute('sensor_tick', str(CameraSettings.sensor_tick))
        camera_bp.set_attribute('shutter_speed', str(CameraSettings.shutter_speed))

        instance_segmentation_camera_bp = blueprint_library.find('sensor.camera.instance_segmentation')
        instance_segmentation_camera_bp.set_attribute('image_size_x', str(CameraSettings.image_size_x))
        instance_segmentation_camera_bp.set_attribute('image_size_y', str(CameraSettings.image_size_y))
        instance_segmentation_camera_bp.set_attribute('sensor_tick', str(CameraSettings.sensor_tick))
        # instance_segmentation_camera_bp.set_attribute('shutter_speed', str(CameraSettings.shutter_speed))

        spawn_points = world.get_map().get_spawn_points()
        spawn_points = spawn_points if (not is_test) else [spawn_points[0]]

        # Travel
        for i_point, (spawn_point) in enumerate(spawn_points):
            model3_actor = world.spawn_actor(model3_bp, spawn_point)
            # if is_autopilot:
            #     model3_actor.set_autopilot(True)
            # time.sleep(3)

            for i_view, (x, y, z, fov) in enumerate(camera_distance_list):
                xyz_list = extend_xyz(x, y, z) if (not is_test) else [[x, y, z]]

                for i_extend, (x, y, z) in enumerate(xyz_list):

                    # sp: spawn_actor    dis: distance
                    save_name = "%s-sp_%04d-dis_%03d-%d" % (
                        'auto' if is_autopilot else 'static', i_point, i_view, i_extend
                    )
                    pitch, yaw, roll = get_rotation_by_center_actor(x, y, z)
                    camera_transform = carla.Transform(
                        location=carla.Location(x=x, y=y, z=z),
                        rotation=carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
                    )
                    camera_bp.set_attribute('fov', str(fov))
                    instance_segmentation_camera_bp.set_attribute('fov', str(fov))
                    camera_actor = world.spawn_actor(
                        camera_bp, camera_transform, attach_to=model3_actor, attachment_type=carla.AttachmentType.Rigid
                    )
                    calibration = np.identity(3)
                    calibration[0, 2] = CameraSettings.image_size_x / 2.0
                    calibration[1, 2] = CameraSettings.image_size_y / 2.0
                    calibration[0, 0] = CameraSettings.image_size_x / (2.0 * np.tan(fov * np.pi / 360.0))
                    calibration[1, 1] = CameraSettings.image_size_y / (2.0 * np.tan(fov * np.pi / 360.0))
                    camera_actor.calibration = calibration

                    instance_segmentation_camera_actor = world.spawn_actor(
                        instance_segmentation_camera_bp,
                        camera_transform,
                        attach_to=model3_actor,
                        attachment_type=carla.AttachmentType.Rigid
                    )

                    time.sleep(1)

                    vehicle_t = model3_actor.get_transform()
                    bounding_boxes = ClientSideBoundingBoxes.get_bounding_box(model3_actor, camera_actor)

                    label_content = {
                        "image": save_name + ".png",
                        "map": world_map,
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
                    }

                    camera_actor.listen(
                        lambda image: save_img(
                            image,                                                               #
                            [CameraSettings.image_size_x, CameraSettings.image_size_y],
                            bounding_boxes,
                            os.path.join(data_save_dir, 'images', world_map, f'{save_name}.png')
                        )
                    )

                    # https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
                    instance_segmentation_camera_actor.listen(
                        lambda image: save_seg_img(
                            image,                                                                      #
                            [CameraSettings.image_size_x, CameraSettings.image_size_y],
                            bounding_boxes,
                            os.path.join(data_save_dir, 'segmentations', world_map, f'{save_name}.png')
                        )
                    )

                    with open(os.path.join(data_save_dir, 'labels', world_map, f'{save_name}.json'), 'w') as f:
                        json.dump(label_content, f, indent=4, ensure_ascii=False)

                    time.sleep(2) # wait for render to world
                    camera_actor.stop()
                    instance_segmentation_camera_actor.stop()
                    carla.command.DestroyActor(camera_actor)
                    carla.command.DestroyActor(instance_segmentation_camera_actor)
            carla.command.DestroyActor(model3_actor)

    finally:
        print(' Finish in %.2f s' % (time.time() - current_time))
        print('... Destroying All Actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])


def save_img(image, image_sizes: List[int], bounding_boxes, save_path):
    # print(image)
    img_raw_bytes = np.array(image.raw_data)
    img_channel_4 = img_raw_bytes.reshape((image_sizes[0], image_sizes[1], 4))
    img_channel3 = img_channel_4[:, :, : 3]
    # img_channel3 = ClientSideBoundingBoxes.draw_bounding_boxes(img_channel3, bounding_boxes)
    if True:
        cv2.imwrite(save_path, img_channel3)
        cv2.waitKey(1)


def save_seg_img(image, image_sizes: List[int], bounding_boxes, save_path):
    # print(image)
    img_raw_bytes = np.array(image.raw_data)
    img_channel_4 = img_raw_bytes.reshape((image_sizes[0], image_sizes[1], 4))
    img_channel3 = img_channel_4[:, :, : 3]
    # img_channel3 = ClientSideBoundingBoxes.draw_bounding_boxes(img_channel3, bounding_boxes)
    if True:
        cv2.imwrite(save_path, img_channel3)
        cv2.waitKey(1)


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


class CameraManager:
    @staticmethod
    def get_camera(
        world,
        type: str,
        vehicle,
        img_size=800,
        fov=90,
        sensor_tick=1.0,
        location=[1, 1, 1],
        attachment_type=carla.AttachmentType.Rigid
    ):
        valid_types = ['rgb', 'instance_segmentation']
        camera_bp = world.get_blueprint_library().find(f'sensor.camera.{type}')
        camera_bp.set_attribute('image_size_x', str(img_size))
        camera_bp.set_attribute('image_size_y', str(img_size))
        camera_bp.set_attribute('fov', str(fov))
        camera_bp.set_attribute('sensor_tick', str(sensor_tick))

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

        x, y, z = location
        pitch, yaw, roll = get_rotation_by_center_actor(x, y, z)
        camera_transform = carla.Transform(
            location=carla.Location(x=x, y=y, z=z), rotation=carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
        )

        camera_actor = world.spawn_actor(
            camera_bp, camera_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid
        )
        calibration = np.identity(3)
        calibration[0, 2] = img_size / 2.0
        calibration[1, 2] = img_size / 2.0
        calibration[0, 0] = calibration[1, 1] = img_size / (2.0 * np.tan(fov * np.pi / 360.0))
        camera_actor.calibration = calibration

        return camera_actor


def extend_xyz(x, y, z):
    return [
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


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    获取 3D bbox   
    """
    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_box = ClientSideBoundingBoxes.get_bounding_box(vehicle, camera)
        # filter objects behind camera
        return bounding_box

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(img, bbox):
        img = img.copy()

        points = np.array([(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)])
        for pt in points:
            cv2.circle(img, pt, 1, (0, 0, 255), 2)
            cv2.putText(img, "(%d,%d)" % (pt[0], pt[1]), (pt[0], pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        pt1 = (np.min(points[:, 0]), np.min(points[:, 1]))
        pt2 = (np.max(points[:, 0]), np.max(points[:, 1]))

        cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
        return img

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[: 3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


if __name__ == '__main__':
    try:
        for world_map in MAPS:
            main(world_map)
    except KeyboardInterrupt:
        pass
    finally:
        print('[Exit].')
