from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from arena_humansim.occlusion.bitmap import _MARGIN, _MIN_STEP, _RESOLUTION, BitmapOccluder
from arena_humansim.utils.benchmark import generate_maze


def _occ() -> BitmapOccluder:
    return BitmapOccluder()


def test_empty_walls_returns_all_true() -> None:
    occ = _occ()
    occ.set_walls([])
    p_a = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, -3.0]])
    p_b = np.array([[1.0, 0.0], [-1.0, 2.0], [10.0, 7.0]])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.ones(3, dtype=np.bool_))


def test_wall_between_two_points_blocks_ray() -> None:
    # Vertical wall at x=0 from y=-1 to y=1 blocks ray from (-2, 0) to (2, 0).
    occ = _occ()
    occ.set_walls([((0.0, -1.0), (0.0, 1.0))])
    p_a = np.array([[-2.0, 0.0]])
    p_b = np.array([[2.0, 0.0]])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.array([False]))


def test_wall_offset_to_side_does_not_block() -> None:
    # Vertical wall at x=5 does not intersect the ray from (0,0) to (3,0).
    occ = _occ()
    occ.set_walls([((5.0, -1.0), (5.0, 1.0))])
    p_a = np.array([[0.0, 0.0]])
    p_b = np.array([[3.0, 0.0]])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.array([True]))


def test_mixed_rays_some_blocked_some_clear() -> None:
    # Horizontal wall at y=0 from x=-1 to x=1.
    occ = _occ()
    occ.set_walls([((-1.0, 0.0), (1.0, 0.0))])
    # ray 0: crosses the wall (from y=-1 to y=1, x=0) -> blocked
    # ray 1: does not cross (from y=2 to y=3, x=0) -> clear
    # ray 2: lateral, wall at y=0 between x=-1..1, ray goes from (3,0) to (5,0); endpoint on wall row is borderline; use a safe side
    # ray 1: fully above, guaranteed clear
    p_a = np.array([[0.0, -1.0], [0.0, 2.0]])
    p_b = np.array([[0.0, 1.0], [0.0, 3.0]])
    result = occ.clear(p_a, p_b)
    assert result[0] == False  # noqa: E712
    assert result[1] == True  # noqa: E712


def test_clear_single_point_pair_no_division_by_zero() -> None:
    # Both endpoints identical - zero-length ray; must return True (no wall hit).
    occ = _occ()
    occ.set_walls([((0.0, -1.0), (0.0, 1.0))])
    p_a = np.array([[0.0, 0.0]])
    p_b = np.array([[0.0, 0.0]])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.array([True]))


def test_set_walls_empty_after_non_empty_resets_to_pass_through() -> None:
    occ = _occ()
    occ.set_walls([((0.0, -1.0), (0.0, 1.0))])
    p_a = np.array([[-2.0, 0.0]])
    p_b = np.array([[2.0, 0.0]])
    assert occ.clear(p_a, p_b)[0] == False  # noqa: E712

    occ.set_walls([])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.array([True]))


def test_wall_perpendicular_ray_not_blocked() -> None:
    # Horizontal wall at y=5; ray travels horizontally at y=0; should not block.
    occ = _occ()
    occ.set_walls([((-10.0, 5.0), (10.0, 5.0))])
    p_a = np.array([[-3.0, 0.0]])
    p_b = np.array([[3.0, 0.0]])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.array([True]))


def test_diagonal_wall_blocks_crossing_ray() -> None:
    # 45deg wall from (-1,-1) to (1,1) blocks a ray from (-1,1) to (1,-1).
    occ = _occ()
    occ.set_walls([((-1.0, -1.0), (1.0, 1.0))])
    p_a = np.array([[-1.0, 1.0]])
    p_b = np.array([[1.0, -1.0]])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.array([False]))


def test_empty_input_returns_empty_array() -> None:
    occ = _occ()
    occ.set_walls([((0.0, 0.0), (0.0, 1.0))])
    p_a = np.empty((0, 2))
    p_b = np.empty((0, 2))
    result = occ.clear(p_a, p_b)
    assert result.shape == (0,)


def _sample_points(occ: BitmapOccluder, p_a: np.ndarray, p_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.hypot(*(p_b - p_a).T)
    n = int(np.ceil(lengths.max() / (_RESOLUTION / 8.0))) + 1
    ts = np.linspace(0.0, 1.0, n)
    px = p_a[:, 0:1] + ts[np.newaxis, :] * (p_b[:, 0] - p_a[:, 0])[:, np.newaxis]
    py = p_a[:, 1:2] + ts[np.newaxis, :] * (p_b[:, 1] - p_a[:, 1])[:, np.newaxis]
    return px, py


def _oracle_clear(occ: BitmapOccluder, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    grid = occ._grid
    assert grid is not None
    height, width = grid.shape
    result = np.ones(len(p_a), dtype=np.bool_)
    for start in range(0, len(p_a), 500):
        sl = slice(start, start + 500)
        px, py = _sample_points(occ, p_a[sl], p_b[sl])
        col = ((px - occ._origin_x) / _RESOLUTION).astype(np.int32)
        row = ((py - occ._origin_y) / _RESOLUTION).astype(np.int32)
        in_bounds = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        occupied = grid[np.clip(row, 0, height - 1), np.clip(col, 0, width - 1)] & in_bounds
        result[sl] = ~occupied.any(axis=1)
    return result


def _maze_occluder() -> BitmapOccluder:
    walls = [((float(x0), float(y0)), (float(x1), float(y1))) for (x0, y0), (x1, y1) in generate_maze(10, seed=42).walls]
    occ = _occ()
    occ.set_walls(walls)
    return occ


def test_march_agrees_with_dense_sampling_on_maze() -> None:
    occ = _maze_occluder()
    rng = np.random.default_rng(7)
    n = 20000
    p_a = rng.uniform(0.0, 20.0, size=(n, 2))
    angle = rng.uniform(0.0, 2.0 * math.pi, size=n)
    length = rng.uniform(0.0, 10.0, size=n)
    p_b = np.clip(p_a + length[:, np.newaxis] * np.column_stack([np.cos(angle), np.sin(angle)]), 0.0, 20.0)

    marched = occ.clear(p_a, p_b)
    oracle = _oracle_clear(occ, p_a, p_b)

    disagree = np.flatnonzero(marched != oracle)
    assert disagree.size / n < 0.005

    grid = occ._grid
    assert grid is not None
    rows, cols = np.nonzero(grid)
    lo_x = occ._origin_x + cols * _RESOLUTION
    lo_y = occ._origin_y + rows * _RESOLUTION
    tree = cKDTree(np.column_stack([lo_x + _RESOLUTION / 2.0, lo_y + _RESOLUTION / 2.0]))
    for i in disagree:
        px, py = _sample_points(occ, p_a[i : i + 1], p_b[i : i + 1])
        pts = np.column_stack([px.ravel(), py.ravel()])
        _, nearest = tree.query(pts)
        gap_x = np.maximum(np.maximum(lo_x[nearest] - pts[:, 0], pts[:, 0] - lo_x[nearest] - _RESOLUTION), 0.0)
        gap_y = np.maximum(np.maximum(lo_y[nearest] - pts[:, 1], pts[:, 1] - lo_y[nearest] - _RESOLUTION), 0.0)
        assert np.hypot(gap_x, gap_y).min() <= _RESOLUTION * math.sqrt(2.0)


@pytest.mark.parametrize("endpoint", ["start", "end"])
def test_endpoint_inside_occupied_cell_blocks_ray(endpoint: str) -> None:
    occ = _occ()
    occ.set_walls([((0.0, -1.0), (0.0, 1.0))])
    on_wall = [-0.04, 0.5]
    free = [-0.06, 0.5]
    grid = occ._grid
    assert grid is not None
    assert grid[30, 19] and not grid[30, 18]
    p_a = np.array([on_wall if endpoint == "start" else free])
    p_b = np.array([free if endpoint == "start" else on_wall])
    np.testing.assert_array_equal(occ.clear(p_a, p_b), np.array([False]))


def _numpy_march(occ: BitmapOccluder, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    result = np.ones(len(p_a), dtype=np.bool_)
    assert occ._dist is not None
    height, width = occ._dist.shape
    dist = occ._dist.ravel()
    ox = occ._origin_x
    oy = occ._origin_y

    dx = p_b[:, 0] - p_a[:, 0]
    dy = p_b[:, 1] - p_a[:, 1]
    lengths = np.hypot(dx, dy)

    idx = np.flatnonzero(lengths > 1e-9)
    ax = p_a[idx, 0]
    ay = p_a[idx, 1]
    dx = dx[idx]
    dy = dy[idx]
    inv_len = 1.0 / lengths[idx]
    s = np.zeros(idx.size, dtype=np.float64)

    while idx.size:
        col = ((ax + s * dx - ox) / _RESOLUTION).astype(np.int32)
        row = ((ay + s * dy - oy) / _RESOLUTION).astype(np.int32)
        in_bounds = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        d = dist[np.clip(row, 0, height - 1) * width + np.clip(col, 0, width - 1)]

        blocked = in_bounds & (d == 0.0)
        result[idx[blocked]] = False

        keep = ~blocked & (s < 1.0)
        s = np.minimum(s + np.maximum(d - _MARGIN, _MIN_STEP) * inv_len, 1.0)
        idx = idx[keep]
        ax = ax[keep]
        ay = ay[keep]
        dx = dx[keep]
        dy = dy[keep]
        inv_len = inv_len[keep]
        s = s[keep]

    return result


def test_kernel_matches_numpy_march_on_maze() -> None:
    occ = _maze_occluder()
    assert occ._dist is not None
    height, width = occ._dist.shape
    rng = np.random.default_rng(11)
    n = 20000
    lo_x = occ._origin_x - 3.0
    lo_y = occ._origin_y - 3.0
    hi_x = occ._origin_x + width * _RESOLUTION + 3.0
    hi_y = occ._origin_y + height * _RESOLUTION + 3.0
    p_a = np.column_stack([rng.uniform(lo_x, hi_x, size=n), rng.uniform(lo_y, hi_y, size=n)])
    angle = rng.uniform(0.0, 2.0 * math.pi, size=n)
    length = rng.uniform(0.0, 15.0, size=n)
    length[: n // 20] = 0.0
    p_b = p_a + length[:, np.newaxis] * np.column_stack([np.cos(angle), np.sin(angle)])

    outside = (p_b[:, 0] < occ._origin_x) | (p_b[:, 0] >= hi_x - 3.0) | (p_b[:, 1] < occ._origin_y) | (p_b[:, 1] >= hi_y - 3.0)
    expected = _numpy_march(occ, p_a, p_b)
    assert outside.sum() > 1000
    assert 0 < expected.sum() < n - n // 20

    np.testing.assert_array_equal(occ.clear(p_a, p_b), expected)
