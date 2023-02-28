import os
import shutil
import time
import json
from typing import List

import cv2
import numpy as np

import carla


class ARGS:
    data_root = 'tmp/data'
    is_test = False
    autopilot = False
    MAPS = [
        'Town01',
        'Town04',
        'Town05',
        'Town06',
        'Town07',
        'Town08',
        'Town09',
    ]
    camera_distance_list = [
                             # [x,y,z,fov]
                             # [3, 2, 2, 100],
                             # [3, 2, 2,00],
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


def main():

    for world_map in ARGS.MAPS:
        current_time = time.time()
        client = None
        actor_list = []
        pilot_type = 'auto' if ARGS.autopilot else 'static'
        try:
            # set up data saving dir
            save_dir_map = {
                name: os.path.join(ARGS.data_root, name, world_map, pilot_type)
                for name in [
                    'images',
                    'segmentations',
                    'labels',
                ]
            }
            for d in save_dir_map.values():
                # if os.path.exists(d):
                #     shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)

            client = carla.Client('localhost', 2000)
            client.set_timeout(60)
            world = client.load_world(world_map)
            print(' Load World(%s) in %.2f s' % (world_map, time.time() - current_time))
            current_time = time.time()

            # Get blueprint_library
            blueprint_library = world.get_blueprint_library()
            with open(os.path.join(ARGS.data_root, 'blueprint_library.json'), 'w') as f:
                bp_list = []
                for index, bp in enumerate(blueprint_library):
                    bp_list.append({"index": index, "id": bp.id, "tags": bp.tags})
                json.dump({"blueprint_library": bp_list}, f, indent=4, ensure_ascii=False)

            # blueprint model3 vehicle
            model3_bp = blueprint_library.find('vehicle.tesla.model3')

            # blueprint camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(ARGS.CameraSettings.image_size_x))
            camera_bp.set_attribute('image_size_y', str(ARGS.CameraSettings.image_size_y))
            camera_bp.set_attribute('sensor_tick', str(ARGS.CameraSettings.sensor_tick))
            camera_bp.set_attribute('shutter_speed', str(ARGS.CameraSettings.shutter_speed))

            instance_segmentation_camera_bp = blueprint_library.find('sensor.camera.instance_segmentation')
            instance_segmentation_camera_bp.set_attribute('image_size_x', str(ARGS.CameraSettings.image_size_x))
            instance_segmentation_camera_bp.set_attribute('image_size_y', str(ARGS.CameraSettings.image_size_y))
            instance_segmentation_camera_bp.set_attribute('sensor_tick', str(ARGS.CameraSettings.sensor_tick))
            # instance_segmentation_camera_bp.set_attribute('shutter_speed', str(CameraSettings.shutter_speed))

            spawn_points = world.get_map().get_spawn_points()
            spawn_points = spawn_points if (not ARGS.is_test) else [spawn_points[0]]

            # Travel
            for i_point, (spawn_point) in enumerate(spawn_points):
                model3_actor = world.spawn_actor(model3_bp, spawn_point)
                time.sleep(3)
                if ARGS.autopilot:
                    model3_actor.set_autopilot(True)

                for i_view, (x, y, z, fov) in enumerate(ARGS.camera_distance_list):
                    xyz_list = Utils.extend_xyz(x, y, z)

                    for i_extend, (x, y, z) in enumerate(xyz_list):
                        print('\033[1;32m [Point] \033[0m',end='')
                        print('SpawnPoint: %-4d'%(i_point),end='')
                        print('View: %-4d'%(i_view),end='')
                        print('Direction: ( %-2.2f, %-2.2f, %-2.2f )'%(x,y,z),end='')
                        print()
                        # sp: spawn_actor    dis: distance
                        save_name = "%s-sp_%04d-dis_%03d-%d" % (pilot_type, i_point, i_view, i_extend)
                        pitch, yaw, roll = Utils.get_rotation_by_center_actor(x, y, z)
                        camera_transform = carla.Transform(
                            location=carla.Location(x=x, y=y, z=z),
                            rotation=carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
                        )
                        camera_bp.set_attribute('fov', str(fov))
                        instance_segmentation_camera_bp.set_attribute('fov', str(fov))
                        camera_actor = world.spawn_actor(
                            camera_bp,
                            camera_transform,
                            attach_to=model3_actor,
                            attachment_type=carla.AttachmentType.Rigid
                        )
                        calibration = np.identity(3)
                        calibration[0, 2] = ARGS.CameraSettings.image_size_x / 2.0
                        calibration[1, 2] = ARGS.CameraSettings.image_size_y / 2.0
                        calibration[0, 0] = ARGS.CameraSettings.image_size_x / (2.0 * np.tan(fov * np.pi / 360.0))
                        calibration[1, 1] = ARGS.CameraSettings.image_size_y / (2.0 * np.tan(fov * np.pi / 360.0))
                        camera_actor.calibration = calibration

                        instance_segmentation_camera_actor = world.spawn_actor(
                            instance_segmentation_camera_bp,
                            camera_transform,
                            attach_to=model3_actor,
                            attachment_type=carla.AttachmentType.Rigid
                        )

                        vehicle_t = model3_actor.get_transform()
                        # bounding_boxes = ClientSideBoundingBoxes.get_bounding_box(model3_actor, camera_actor)

                        camera_actor.listen(
                            lambda image: Utils.save_img(
                                image,                                                                #
                                [ARGS.CameraSettings.image_size_x, ARGS.CameraSettings.image_size_y],
                                os.path.join(save_dir_map['images'], f'{save_name}.png')
                            )
                        )

                        # https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
                        instance_segmentation_camera_actor.listen(
                            lambda image: Utils.save_seg_img(
                                image,                                                                #
                                [ARGS.CameraSettings.image_size_x, ARGS.CameraSettings.image_size_y],
                                os.path.join(save_dir_map['segmentations'], f'{save_name}.png')
                            )
                        )

                        with open(os.path.join(save_dir_map['labels'], f'{save_name}.json'), 'w') as f_label:
                            json.dump(
                                {
                                    "image": save_name + ".png",
                                    "map": world_map,
                                    "pilot_type":pilot_type,
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

                        time.sleep(3) # wait for retrieving data

                        camera_actor.stop()
                        instance_segmentation_camera_actor.stop()

                        carla.command.DestroyActor(camera_actor)
                        carla.command.DestroyActor(instance_segmentation_camera_actor)
                carla.command.DestroyActor(model3_actor)

        except RuntimeError as e:
            print(e)
        finally:
            print(' Finish in %.2f s' % (time.time() - current_time))
            print('... Destroying All Actors')
            if client:
                client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        
        # sleep for next map
        time.sleep(30)


class Utils:
    @staticmethod
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
