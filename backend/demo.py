"""Synthetic BEV observations; validates fusion/planning, not RGB-D hardware."""

import numpy as np

from . import config
from .processing.bev import BEVResult, fuse_results


def make_demo_scene():
    h, w = config.BEV_SIZE
    x = config.BEV_X_RANGE[0] + (np.arange(w) + 0.5) / w * np.ptp(config.BEV_X_RANGE)
    z = config.BEV_Z_RANGE[0] + (h - np.arange(h) - 0.5) / h * np.ptp(
        config.BEV_Z_RANGE
    )
    xx, zz = np.meshgrid(x, z)
    angle = np.degrees(np.arctan2(xx, zz))
    obstacle = (abs(xx) < 0.5) & (zz > 2.0) & (zz < 2.8)
    observed = (zz < 4.8) & (abs(angle) < 78)
    fov = observed.copy()
    observed &= ~((xx < -1.3) & (zz > 3.4))
    cameras = []
    for yaw in (-45, 0, 45):
        occupied = observed & (abs(angle - yaw) < 34)
        low = np.where(occupied, np.where(obstacle, -0.6, 0.0), 999.0)
        high = np.where(occupied, 0.0, -999.0)
        color = np.empty((h, w, 3), np.uint8)
        color[:] = (90, 110, 115)
        color[obstacle] = (215, 130, 65)
        color[~occupied] = 20
        cameras.append(
            BEVResult(
                color_image=color,
                occupied=occupied,
                min_height=low,
                max_height=high,
                height_range=np.where(occupied, high - low, 0),
                count=occupied.astype(float),
                fx=500,
                bev_h=h,
                bev_w=w,
                x_range=config.BEV_X_RANGE,
                z_range=config.BEV_Z_RANGE,
            )
        )
    return fuse_results(cameras), fov
