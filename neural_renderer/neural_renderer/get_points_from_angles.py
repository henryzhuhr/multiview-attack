from __future__ import division
import math

import torch


def get_points_from_angles(
    distance,
    elevation,              # 抬高 海拔
    azimuth,                # 方位
    degrees=True
):
    """
        - `distance`  距离
        - `elevation` 抬高 海拔
        - `azimuth`   方位
    """
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = math.radians(elevation)
            azimuth = math.radians(azimuth)
        return (
            distance * math.cos(elevation) * math.sin(azimuth), distance * math.sin(elevation),
            -distance * math.cos(elevation) * math.cos(azimuth)
        )
    else:
        if degrees:
            elevation = math.pi / 180. * elevation
            azimuth = math.pi / 180. * azimuth
            #
        return torch.stack(
            [
                distance * torch.cos(elevation) * torch.sin(azimuth), 
                distance * torch.sin(elevation),
                -distance * torch.cos(elevation) * torch.cos(azimuth)
            ]
        ).transpose(1, 0)
