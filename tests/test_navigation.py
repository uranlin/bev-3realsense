import numpy as np
import pytest

from backend import config
from backend.processing.bev import BEVProjection, fuse_results
from backend.processing.planner import (
    compute_costmap,
    inflate_obstacles,
    path_grid_to_world,
    plan_path,
    world_to_grid,
)
from backend.utils.camera import make_intrinsic, scale_intrinsic


def test_unknown_wall_is_not_traversable():
    obs = np.zeros((7, 7), bool)
    known = ~obs
    known[:, 3] = False
    assert plan_path(obs, known, (3, 1), (3, 5)) is None


def test_diagonal_corner_cut_is_blocked():
    obs = np.array([[False, True], [True, False]])
    assert plan_path(obs, np.ones_like(obs), (0, 0), (1, 1)) is None


def test_infinite_cost_and_invalid_endpoints_are_blocked():
    obs = np.zeros((5, 5), bool)
    cost = np.ones((5, 5))
    cost[:, 2] = np.inf
    assert plan_path(obs, ~obs, (2, 0), (2, 4), cost) is None
    assert plan_path(obs, ~obs, (-1, 0), (2, 4)) is None
    obs[2, 4] = True
    assert plan_path(obs, ~obs, (2, 0), (2, 4)) is None


def test_small_grid_search_finishes():
    obs = np.zeros((2, 2), bool)
    assert plan_path(obs, ~obs, (0, 0), (1, 1)) == [(0, 0), (1, 1)]
    assert plan_path(obs, ~obs, (0, 0), (0, 0)) == [(0, 0)]


def test_inflation_uses_metric_anisotropic_spacing(monkeypatch):
    monkeypatch.setattr(config, "BEV_X_RANGE", (0, 10))
    monkeypatch.setattr(config, "BEV_Z_RANGE", (0, 5))
    obs = np.zeros((5, 5), bool)
    obs[2, 2] = True
    inflated = inflate_obstacles(obs, 1.1)
    assert inflated[1, 2] and not inflated[2, 1]
    assert not inflate_obstacles(np.zeros_like(obs), 2).any()


def test_grid_roundtrip_and_outside_rejected():
    cells = [(0, 0), (250, 250), (499, 499)]
    assert [world_to_grid(*p) for p in path_grid_to_world(cells)] == cells
    with pytest.raises(ValueError):
        world_to_grid(999, 2)


def test_intrinsics_zero_principal_point_and_resize():
    k = make_intrinsic(100, 120, 0, 0)
    scaled = scale_intrinsic(k, (100, 100), (200, 300))
    np.testing.assert_allclose(np.diag(scaled), [200, 360, 1])
    assert scaled[0, 2] == 0


def test_projection_ignores_invalid_depth_and_negative_boundary(monkeypatch):
    monkeypatch.setattr(config, "BEV_SIZE", (10, 10))
    monkeypatch.setattr(config, "BEV_X_RANGE", (0, 2))
    monkeypatch.setattr(config, "BEV_Z_RANGE", (0, 2))
    depth = np.array([[0, np.nan, np.inf, -1, 1, 10]], np.float32)
    k = make_intrinsic(100, 100, 4, 0)
    bev = BEVProjection().project(np.zeros((1, 6, 3), np.uint8), depth, k, np.eye(4))
    assert bev.count.sum() == 1
    e = np.eye(4)
    e[0, 3] = -0.01
    bev = BEVProjection().project(np.zeros((1, 6, 3), np.uint8), depth, k, e)
    assert bev.count.sum() == 0


def test_fusion_preserves_observations_and_counts():
    from backend.demo import make_demo_scene

    bev, _ = make_demo_scene()
    empty = BEVProjection().project(
        np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1)), np.eye(3), np.eye(4)
    )
    merged = fuse_results([bev, empty])
    np.testing.assert_array_equal(merged.color_image, bev.color_image)
    np.testing.assert_array_equal(merged.count, bev.count)
    doubled = fuse_results([bev, bev])
    np.testing.assert_array_equal(doubled.count, bev.count * 2)


def test_costmap_unknown_cells_are_infinite():
    obs = np.zeros((3, 3), bool)
    known = ~obs
    known[0, 0] = False
    assert np.isinf(compute_costmap(obs, known)[0, 0])
