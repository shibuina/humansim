from __future__ import annotations

import numpy as np
import pytest

from arena_humansim.utils.benchmark import generate_maze
from arena_humansim.utils.wall_grid import WallGrid, query_walls


def _query(grid: WallGrid, x: float, y: float, radius: float, size: int = 4096) -> np.ndarray:
    out = np.empty(size, dtype=np.int64)
    k = query_walls(grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny, x, y, radius, out)
    assert k <= size
    return out[:k]


def _segment_distances(seg: np.ndarray, x: float, y: float) -> np.ndarray:
    a = seg[:, :2]
    d = seg[:, 2:] - a
    span = np.maximum((d * d).sum(axis=1), 1e-12)
    t = np.clip(((x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]) / span, 0.0, 1.0)
    return np.hypot(x - a[:, 0] - t * d[:, 0], y - a[:, 1] - t * d[:, 1])


def _walls(seed: int) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    maze = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in generate_maze(10, seed=42).walls]
    long_diagonals = [((float(x0), float(y0)), (float(x1), float(y1))) for x0, y0, x1, y1 in rng.uniform(-5.0, 25.0, size=(30, 4))]
    return maze + long_diagonals + [((3.0, 3.0), (3.0, 3.0))]


@pytest.mark.parametrize("cell", [0.5, 1.0, 3.0])
def test_query_returns_every_wall_within_radius_once_in_ascending_order(cell: float) -> None:
    walls = _walls(0)
    grid = WallGrid(walls, cell=cell)
    rng = np.random.default_rng(1)
    for x, y, r in zip(rng.uniform(-8.0, 28.0, 400), rng.uniform(-8.0, 28.0, 400), rng.uniform(0.0, 4.0, 400), strict=True):
        found = _query(grid, float(x), float(y), float(r))
        within = np.flatnonzero(_segment_distances(grid.segments, float(x), float(y)) <= r)
        assert np.all(np.diff(found) > 0)
        assert set(within.tolist()) <= set(found.tolist())


def test_query_far_outside_the_grid_finds_nothing() -> None:
    grid = WallGrid(_walls(2))
    assert len(_query(grid, 500.0, -500.0, 3.0)) == 0


def test_empty_grid_finds_nothing() -> None:
    grid = WallGrid([])
    assert len(_query(grid, 0.0, 0.0, 10.0)) == 0


def test_overflowing_buffer_reports_full_count() -> None:
    grid = WallGrid(_walls(3))
    full = _query(grid, 10.0, 10.0, 6.0)
    out = np.empty(2, dtype=np.int64)
    k = query_walls(grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny, 10.0, 10.0, 6.0, out)
    assert k == len(full) > 2
