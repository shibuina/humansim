from __future__ import annotations

import numpy as np
import pytest

from arena_humansim.local_planner import LocalPlanner
from arena_humansim.local_planner.hsfm import HSFMPlanner
from arena_humansim.local_planner.sfm import SFMPlanner
from arena_humansim.utils.benchmark import generate_maze
from arena_humansim.utils.types import Segments

_EPS = 1e-6


def _maze_walls() -> Segments:
    walls: Segments = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in generate_maze(10, seed=42).walls]
    walls += [((0.0, 0.0), (20.0, 20.0)), ((-1.0, 19.0), (19.0, -1.0)), ((3.3, 17.1), (16.2, 2.4)), ((5.0, 5.0), (5.0, 5.0))]
    return walls


def _dense_reference(walls: Segments, pos: np.ndarray, radius: np.ndarray, strength: float, rng: float) -> np.ndarray:
    seg = np.array(walls, dtype=np.float64).reshape(-1, 2, 2)
    seg_p1 = seg[:, 0, :]
    seg_d = seg[:, 1, :] - seg_p1
    seg_len_sq = np.sum(seg_d**2, axis=1)
    ap = pos[:, None, :]
    diff_to_p1 = ap - seg_p1[None, :, :]
    t = np.sum(diff_to_p1 * seg_d[None, :, :], axis=2) / np.maximum(seg_len_sq[None, :], _EPS)
    t = np.clip(t, 0.0, 1.0)
    cp = seg_p1[None, :, :] + t[:, :, None] * seg_d[None, :, :]
    diff = ap - cp
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    dist = np.maximum(dist, _EPS)
    normals = diff / dist[:, :, None]
    mag = strength * np.exp((radius[:, None] - dist) / rng)
    return (mag[:, :, None] * normals).sum(axis=1)


def _agents(walls: Segments) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    seg = np.array(walls, dtype=np.float64).reshape(-1, 4)
    lo = seg.reshape(-1, 2).min(axis=0) - 1.0
    hi = seg.reshape(-1, 2).max(axis=0) + 1.0
    scattered = rng.uniform(lo, hi, size=(400, 2))
    pick = rng.integers(0, len(seg), size=200)
    t = rng.uniform(0.0, 1.0, size=200)[:, None]
    on_wall = seg[pick, :2] + t * (seg[pick, 2:] - seg[pick, :2]) + rng.normal(0.0, 0.15, size=(200, 2))
    pos = np.vstack([scattered, on_wall, seg[:5, :2]])
    radius = rng.uniform(0.15, 0.45, size=len(pos))
    return pos, radius


@pytest.mark.parametrize(
    "make",
    [
        lambda: LocalPlanner.create("sfm"),
        lambda: LocalPlanner.create("hsfm"),
        lambda: SFMPlanner(wall_repulsion_strength=5.0, wall_repulsion_range=0.25),
        lambda: HSFMPlanner(wall_repulsion_strength=1.5, wall_repulsion_range=0.4),
    ],
    ids=["sfm", "hsfm", "sfm_custom", "hsfm_custom"],
)
def test_grid_wall_forces_match_dense_reference(make) -> None:
    planner = make()
    assert isinstance(planner, SFMPlanner)
    walls = _maze_walls()
    planner.set_walls(walls)
    pos, radius = _agents(walls)
    got = planner._compute_wall_forces_vectorized(pos, radius)
    ref = _dense_reference(walls, pos, radius, planner.wall_repulsion_strength, planner.wall_repulsion_range)
    assert got.shape == (len(pos), 2)
    assert np.max(np.abs(got - ref)) <= 1e-8


def test_agent_far_from_walls_gets_zero_wall_force() -> None:
    planner = SFMPlanner()
    planner.set_walls(_maze_walls())
    got = planner._compute_wall_forces_vectorized(np.array([[100.0, 100.0], [-50.0, 3.0]]), np.array([0.3, 0.3]))
    assert np.array_equal(got, np.zeros((2, 2)))


def test_no_walls_gives_zero_wall_force() -> None:
    planner = SFMPlanner()
    planner.set_walls([])
    got = planner._compute_wall_forces_vectorized(np.array([[0.0, 0.0], [1.0, 2.0]]), np.array([0.3, 0.3]))
    assert np.array_equal(got, np.zeros((2, 2)))
