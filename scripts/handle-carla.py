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

tm_port = 8000


def main():
    actor_list: List[types.carla.Actor] = []
    vehicle_bps = []
    walkers_list = []
    camera: types.carla.Actor = None
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()

        traffic_manager = client.get_trafficmanager(tm_port)        # 获取交通管理器
        traffic_manager.set_global_distance_to_leading_vehicle(2.5) # 与前车距离

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

        for bp in vehicle_bps:
            shuffled_list = spawn_points.copy()
            random.shuffle(shuffled_list)
            for transform in shuffled_list:
                actor = world.try_spawn_actor(bp, transform)
                if actor is not None:
                    actor_list.append(actor)
                    actor.set_autopilot(True)
                    print('created %s' % actor.type_id)
                    break

        while True:
            world.wait_for_tick()
    finally:
        print('destroying actors')
        if camera:
            camera.destroy()
        print('destroy actors: ', len(actor_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])


if __name__ == '__main__':
    main()
