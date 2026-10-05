from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from arena_humansim.collision import CollisionResolver
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.types import Segments
from arena_humansim.utils.wall_grid import WallGrid, query_walls

_QUERY_CAPACITY = 64


@njit(cache=True)
def _contact(segments: np.ndarray, w: int, x: float, y: float) -> tuple[float, float, float]:
    ax = segments[w, 0]
    ay = segments[w, 1]
    abx = segments[w, 2] - ax
    aby = segments[w, 3] - ay
    ab_sq = max(abx * abx + aby * aby, 1e-12)
    t = min(max(((x - ax) * abx + (y - ay) * aby) / ab_sq, 0.0), 1.0)
    dx = x - (ax + t * abx)
    dy = y - (ay + t * aby)
    dist = math.sqrt(dx * dx + dy * dy)
    safe_dist = dist if dist > 1e-9 else 1e-9
    return dist, dx / safe_dist, dy / safe_dist


@njit(cache=True, parallel=True)
def _resolve_kernel(
    pos: np.ndarray,
    vel: np.ndarray,
    radius: np.ndarray,
    segments: np.ndarray,
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    margin: float,
    corrected: np.ndarray,
) -> None:
    for i in prange(pos.shape[0]):
        corrected[i] = False
        threshold = radius[i] + margin
        buf = np.empty(_QUERY_CAPACITY, dtype=np.int64)
        for _ in range(3):
            x = pos[i, 0]
            y = pos[i, 1]
            k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, threshold, buf)
            if k > buf.shape[0]:
                buf = np.empty(k, dtype=np.int64)
                k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, threshold, buf)

            hit = False
            cx = 0.0
            cy = 0.0
            for c in range(k):
                dist, ux, uy = _contact(segments, buf[c], x, y)
                if dist < threshold - 1e-9:
                    hit = True
                    cx += ux * (threshold - dist)
                    cy += uy * (threshold - dist)
            if not hit:
                break

            corrected[i] = True
            pos[i, 0] = x + cx
            pos[i, 1] = y + cy

            vx = vel[i, 0]
            vy = vel[i, 1]
            for c in range(k):
                dist, ux, uy = _contact(segments, buf[c], x, y)
                if dist < threshold - 1e-9:
                    proj = vx * ux + vy * uy
                    if proj < 0:
                        vx -= proj * ux
                        vy -= proj * uy
            vel[i, 0] = vx
            vel[i, 1] = vy


def _resolve_grid(grid: WallGrid, pos: np.ndarray, vel: np.ndarray, radius: np.ndarray, margin: float) -> np.ndarray:
    corrected = np.empty(pos.shape[0], dtype=np.bool_)
    _resolve_kernel(pos, vel, radius, grid.segments, grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny, margin, corrected)
    return corrected


class WallProjectionResolver(CollisionResolver):
    def __init__(self, margin: float = 0.01):
        self._margin = margin
        self._grid = WallGrid([])
        self._warmup()

    def _warmup(self) -> None:
        pos = np.array([[0.1, 0.1]], dtype=np.float64)
        vel = np.array([[-1.0, -1.0]], dtype=np.float64)
        radius = np.array([0.3], dtype=np.float64)
        _resolve_grid(WallGrid([((0.0, 0.0), (1.0, 0.0))]), pos, vel, radius, self._margin)

    def set_walls(self, segments: Segments) -> None:
        self._grid = WallGrid(segments)

    def resolve(self, pool: AgentPool) -> set[int]:
        n = pool.n
        if n == 0 or self._grid.segments.shape[0] == 0:
            return set()

        corrected = _resolve_grid(self._grid, pool.pos[:n], pool.vel[:n], pool.agent_radius[:n], self._margin)

        if not corrected.any():
            return set()
        return {int(aid) for aid in pool.agent_ids[:n][corrected]}
