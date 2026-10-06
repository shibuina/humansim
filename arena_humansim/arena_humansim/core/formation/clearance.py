from __future__ import annotations

import math
from collections.abc import Collection, Sequence

import numpy as np
from numba import njit

from arena_humansim.core.pool import AgentPool, PoolAware
from arena_humansim.utils.types import Segments, WallAware
from arena_humansim.utils.wall_grid import WallGrid, cast_one, cast_rays


@njit(cache=True)
def _crowd_kernel(
    rows: np.ndarray,
    centers: np.ndarray,
    r2: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    pos: np.ndarray,
    member_ptr: np.ndarray,
    member_rows: np.ndarray,
    out: np.ndarray,
) -> None:
    for g in range(rows.shape[0]):
        i = rows[g]
        count = 0
        if 0 <= i and i + 1 < indptr.shape[0]:
            for t in range(indptr[i], indptr[i + 1]):
                j = indices[t]
                member = False
                for m in range(member_ptr[g], member_ptr[g + 1]):
                    if member_rows[m] == j:
                        member = True
                        break
                if member:
                    continue
                dx = pos[j, 0] - centers[g, 0]
                dy = pos[j, 1] - centers[g, 1]
                if dx * dx + dy * dy < r2[g]:
                    count += 1
        out[g] = count


@njit(cache=True)
def _sides_kernel(
    poses: np.ndarray,
    stations: np.ndarray,
    seg: np.ndarray,
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    out: np.ndarray,
) -> None:
    buf = np.empty(seg.shape[0], dtype=np.int64)
    for g in range(poses.shape[0]):
        x, y, reach = poses[g, 0], poses[g, 1], poses[g, 3]
        hx, hy = math.cos(poses[g, 2]), math.sin(poses[g, 2])
        for side in range(2):
            sign = 1.0 - 2.0 * side
            free = reach
            if seg.shape[0]:
                for s in stations:
                    free = min(free, cast_one(x + s * hx, y + s * hy, -sign * hy, sign * hx, reach, seg, cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, buf))
            out[g, side] = free


class Clearance(WallAware, PoolAware):
    """Free room to the walls along rays, and crowding among an observer's perceived agents."""

    def __init__(self) -> None:
        self._grid = WallGrid([])
        self._pool: AgentPool | None = None
        self._warmup()

    def _warmup(self) -> None:
        grid = WallGrid([((0.0, 1.0), (1.0, 1.0))])
        _sides_kernel(np.zeros((1, 5)), np.zeros(1), grid.segments, grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny, np.empty((1, 2)))
        _crowd_kernel(np.zeros(1, dtype=np.int64), np.zeros((1, 2)), np.ones(1), np.zeros(2, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((1, 2)), np.zeros(2, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64))

    def set_walls(self, segments: Segments) -> None:
        self._grid = WallGrid(segments)

    def attach(self, pool: AgentPool) -> None:
        self._pool = pool

    def free(self, origins: np.ndarray, dirs: np.ndarray, max_dist: float | np.ndarray) -> np.ndarray:
        return cast_rays(self._grid, origins, dirs, max_dist)

    def sides(self, poses: np.ndarray, stations: np.ndarray) -> np.ndarray:
        """Free room left and right of each (x, y, heading, reach) row, the least over rays cast from the given distances ahead."""
        grid = self._grid
        out = np.empty((len(poses), 2))
        _sides_kernel(np.ascontiguousarray(poses, dtype=np.float64), np.ascontiguousarray(stations, dtype=np.float64), grid.segments, grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny, out)
        return out

    def crowds(self, observers: Sequence[int], centers: np.ndarray, radii: float | np.ndarray, exclude: Sequence[Collection[int]]) -> np.ndarray:
        out = np.zeros(len(observers), dtype=np.int64)
        pool = self._pool
        if pool is None or not len(observers):
            return out
        lookup = pool._id_to_idx
        rows = np.array([lookup.get(aid, -1) for aid in observers], dtype=np.int64)
        members = [[lookup[aid] for aid in group if aid in lookup] for group in exclude]
        member_ptr = np.zeros(len(members) + 1, dtype=np.int64)
        np.cumsum([len(m) for m in members], out=member_ptr[1:])
        member_rows = np.array([row for m in members for row in m], dtype=np.int64)
        _crowd_kernel(rows, np.ascontiguousarray(centers, dtype=np.float64), np.broadcast_to(np.square(radii, dtype=np.float64), (len(observers),)).copy(), pool.neighbor_indptr.astype(np.int64), pool.neighbor_indices.astype(np.int64), pool.pos, member_ptr, member_rows, out)
        return out

    def oncoming(self, observer: int, x: float, y: float, radius: float, heading: float, exclude: Collection[int]) -> tuple[int, int]:
        rows = self._near(observer, x, y, radius, exclude)
        if self._pool is None or not len(rows):
            return 0, 0
        pos, vel = self._pool.pos[rows], self._pool.vel[rows]
        hx, hy = np.cos(heading), np.sin(heading)
        against = vel[:, 0] * hx + vel[:, 1] * hy < 0.0
        left = (pos[:, 0] - x) * -hy + (pos[:, 1] - y) * hx > 0.0
        return int(np.count_nonzero(against & left)), int(np.count_nonzero(against & ~left))

    def _near(self, observer: int, x: float, y: float, radius: float, exclude: Collection[int]) -> np.ndarray:
        pool = self._pool
        if pool is None or observer not in pool._id_to_idx:
            return np.zeros(0, dtype=np.int64)
        i = pool.idx(observer)
        if i + 1 >= len(pool.neighbor_indptr):
            return np.zeros(0, dtype=np.int64)
        rows = pool.neighbor_indices[pool.neighbor_indptr[i] : pool.neighbor_indptr[i + 1]]
        ids = pool.agent_ids[rows]
        keep = np.ones(len(rows), dtype=bool)
        for aid in exclude:
            keep &= ids != aid
        rows = rows[keep]
        pos = pool.pos[rows]
        return rows[(pos[:, 0] - x) ** 2 + (pos[:, 1] - y) ** 2 < radius * radius]
