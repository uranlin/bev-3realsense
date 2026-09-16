"""Metric grid planning. Unknown cells are never traversable."""

import heapq
import math

import numpy as np
from scipy import ndimage

from .. import config


def _spacing(shape):
    return (
        (config.BEV_Z_RANGE[1] - config.BEV_Z_RANGE[0]) / shape[0],
        (config.BEV_X_RANGE[1] - config.BEV_X_RANGE[0]) / shape[1],
    )


def inflate_obstacles(obstacle_mask, robot_radius_m=None):
    mask = np.asarray(obstacle_mask, dtype=bool)
    radius = config.ROBOT_RADIUS_M if robot_radius_m is None else robot_radius_m
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Robot radius must be finite and nonnegative")
    if not mask.any() or radius == 0:
        return mask.copy()
    return (
        ndimage.distance_transform_edt(~mask, sampling=_spacing(mask.shape)) <= radius
    )


def compute_costmap(obstacle_mask, fov_mask, max_dist_m=3.0):
    if obstacle_mask.shape != fov_mask.shape or max_dist_m <= 0:
        raise ValueError("Masks must match and distance scale must be positive")
    free = np.asarray(fov_mask, bool) & ~np.asarray(obstacle_mask, bool)
    # Include unknown cells and grid boundaries in clearance calculations.
    distance = ndimage.distance_transform_edt(
        np.pad(free, 1), sampling=_spacing(free.shape)
    )[1:-1, 1:-1]
    cost = np.full(free.shape, np.inf, dtype=np.float32)
    cost[free] = 1.0 + 5.0 * np.exp(-distance[free] / max_dist_m)
    return cost


def plan_path(inflated_mask, fov_mask, start_rc, goal_rc, costmap=None):
    """Return exact endpoints or None; no snapping or diagonal corner cutting.

    fov_mask is the observed traversable domain, NOT merely the camera cone.
    """
    if inflated_mask.shape != fov_mask.shape:
        raise ValueError("Planning masks must have the same shape")
    h, w = inflated_mask.shape
    free = np.asarray(fov_mask, bool) & ~np.asarray(inflated_mask, bool)
    cost = np.ones((h, w)) if costmap is None else np.asarray(costmap)
    if cost.shape != free.shape:
        raise ValueError("Costmap shape does not match masks")
    free &= np.isfinite(cost) & (cost > 0)
    start, goal = tuple(map(int, start_rc)), tuple(map(int, goal_rc))
    for r, c in (start, goal):
        if not (0 <= r < h and 0 <= c < w) or not free[r, c]:
            return None
    dz, dx = _spacing(free.shape)
    min_cost = float(cost[free].min())

    def heuristic(p):
        return math.hypot((p[0] - goal[0]) * dz, (p[1] - goal[1]) * dx) * min_cost

    queue = [(heuristic(start), 0.0, start)]
    scores, previous = {start: 0.0}, {}
    while queue:
        _, distance, current = heapq.heappop(queue)
        if distance > scores[current]:
            continue
        if current == goal:
            path = [current]
            while current != start:
                current = previous[current]
                path.append(current)
            return path[::-1]
        r, c = current
        for dr, dc in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not free[nr, nc]:
                continue
            if dr and dc and (not free[r, nc] or not free[nr, c]):
                continue
            candidate = distance + math.hypot(dr * dz, dc * dx) * float(cost[nr, nc])
            neighbor = (nr, nc)
            if candidate < scores.get(neighbor, math.inf):
                scores[neighbor], previous[neighbor] = candidate, (r, c)
                heapq.heappush(
                    queue, (candidate + heuristic(neighbor), candidate, neighbor)
                )
    return None


def path_grid_to_world(path_cells):
    """Cell-centre coordinates, without rounding away grid precision."""
    dz, dx = _spacing(config.BEV_SIZE)
    return [
        (
            config.BEV_X_RANGE[0] + (c + 0.5) * dx,
            config.BEV_Z_RANGE[0] + (config.BEV_SIZE[0] - r - 0.5) * dz,
        )
        for r, c in path_cells
    ]


def world_to_grid(x_world, z_world):
    xmin, xmax = config.BEV_X_RANGE
    zmin, zmax = config.BEV_Z_RANGE
    if not (
        np.isfinite(x_world)
        and np.isfinite(z_world)
        and xmin <= x_world < xmax
        and zmin <= z_world < zmax
    ):
        raise ValueError("Goal is outside the BEV map")
    dz, dx = _spacing(config.BEV_SIZE)
    return config.BEV_SIZE[0] - 1 - int((z_world - zmin) / dz), int(
        (x_world - xmin) / dx
    )
