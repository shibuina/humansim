"""Helbing, Farkas, Vicsek 2000 social force with body compression and sliding friction, after Menge's AgtHelbing."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from numba import njit, prange

from arena_humansim.core.agents.types import ParamDist

from .force import GOAL, SOCIAL, WALL, ForcePlanner, WallGridArgs, near_walls, wall_point

_EPS = 1e-6
_WALL_CUTOFF_RANGES = 28.0

_TAU, _MASS, _A, _A_WALL, _B, _K, _KAPPA = range(7)


@njit(cache=True, parallel=True)
def _helbing_kernel(
    pos: np.ndarray,
    vel: np.ndarray,
    theta: np.ndarray,
    radius: np.ndarray,
    pref: np.ndarray,
    max_acc: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    params: np.ndarray,
    seg: np.ndarray,
    grid: WallGridArgs,
    dt: float,
    out: np.ndarray,
) -> None:
    for i in prange(pos.shape[0]):
        x = pos[i, 0]
        y = pos[i, 1]
        vx = vel[i, 0]
        vy = vel[i, 1]
        r = radius[i]
        tau = params[i, _TAU]
        mass = params[i, _MASS]
        a = params[i, _A]
        a_wall = params[i, _A_WALL]
        b = params[i, _B]
        k = params[i, _K]
        kappa = params[i, _KAPPA]

        out[i, GOAL, 0] = (pref[i, 0] - vx) / tau
        out[i, GOAL, 1] = (pref[i, 1] - vy) / tau

        fx = 0.0
        fy = 0.0
        for t in range(indptr[i], indptr[i + 1]):
            j = indices[t]
            ox = x - pos[j, 0]
            oy = y - pos[j, 1]
            d = math.hypot(ox, oy)
            if d < _EPS:
                continue
            nx = ox / d
            ny = oy / d
            r_ij = r + radius[j]
            mag = a * math.exp((r_ij - d) / b)
            fx += mag * nx
            fy += mag * ny
            if d < r_ij:
                g = r_ij - d
                tx = ny
                ty = -nx
                dv_t = (vel[j, 0] - vx) * tx + (vel[j, 1] - vy) * ty
                fx += k * g * nx + kappa * g * dv_t * tx
                fy += k * g * ny + kappa * g * dv_t * ty
        out[i, SOCIAL, 0] = fx / mass
        out[i, SOCIAL, 1] = fy / mass

        fx = 0.0
        fy = 0.0
        for w in near_walls(grid, x, y, r + b * _WALL_CUTOFF_RANGES):
            px, py = wall_point(seg, w, x, y)
            ox = x - px
            oy = y - py
            d = math.hypot(ox, oy)
            if d < _EPS:
                continue
            nx = ox / d
            ny = oy / d
            mag = a_wall * math.exp((r - d) / b)
            fx += mag * nx
            fy += mag * ny
            if d < r:
                g = r - d
                v_t = vx * ny - vy * nx
                fx += k * g * nx - kappa * g * v_t * ny
                fy += k * g * ny + kappa * g * v_t * nx
        out[i, WALL, 0] = fx / mass
        out[i, WALL, 1] = fy / mass


class HelbingPlanner(ForcePlanner):
    PARAM_DEFAULTS: ClassVar[dict[str, ParamDist]] = {
        "relaxation_time": ParamDist(0.5, 0.0, clip_low=0.05),
        "mass": ParamDist(80.0, 0.0, clip_low=1.0),
        "agent_scale": ParamDist(2000.0, 0.0, clip_low=0.0),
        "obstacle_scale": ParamDist(2000.0, 0.0, clip_low=0.0),
        "force_distance": ParamDist(0.08, 0.0, clip_low=0.01),
        "body_force": ParamDist(1.2e5, 0.0, clip_low=0.0),
        "friction": ParamDist(2.4e5, 0.0, clip_low=0.0),
    }

    _kernel = staticmethod(_helbing_kernel)
