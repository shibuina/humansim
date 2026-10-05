"""Johansson, Helbing, Shukla 2007 elliptical social force with the neighbor's stride offset, after Menge's AgtJohansson."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from numba import njit, prange

from arena_humansim.core.agents.types import ParamDist

from .force import GOAL, SOCIAL, WALL, ForcePlanner, WallGridArgs, near_walls, wall_point

_EPS = 1e-6
_NEIGHBOR_DIST = 5.0

_TAU, _A, _A_WALL, _B, _STRIDE, _W = range(6)


@njit(cache=True, parallel=True)
def _johansson_kernel(
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
        tau = params[i, _TAU]
        a = params[i, _A]
        a_wall = params[i, _A_WALL]
        b_range = params[i, _B]
        stride = params[i, _STRIDE]
        w = params[i, _W]
        ex = math.cos(theta[i])
        ey = math.sin(theta[i])

        out[i, GOAL, 0] = (pref[i, 0] - vel[i, 0]) / tau
        out[i, GOAL, 1] = (pref[i, 1] - vel[i, 1]) / tau

        fx = 0.0
        fy = 0.0
        for t in range(indptr[i], indptr[i + 1]):
            j = indices[t]
            rx = x - pos[j, 0]
            ry = y - pos[j, 1]
            d = math.sqrt(rx * rx + ry * ry)
            if d < _EPS:
                continue
            sx = vel[j, 0] * stride
            sy = vel[j, 1] * stride
            ox = rx - sx
            oy = ry - sy
            od = math.sqrt(ox * ox + oy * oy)
            if od < _EPS:
                continue
            term1 = d + od
            b = 0.5 * math.sqrt(max(term1 * term1 - (sx * sx + sy * sy), 0.0))
            if b < _EPS:
                continue
            nx = rx / d
            ny = ry / d
            cos_theta = nx * ex + ny * ey
            mag = a * (w + (1.0 - w) * (1.0 - cos_theta) * 0.5) * term1 / (2.0 * b) * math.exp(-b / b_range)
            fx += mag * 0.5 * (nx + ox / od)
            fy += mag * 0.5 * (ny + oy / od)
        out[i, SOCIAL, 0] = fx
        out[i, SOCIAL, 1] = fy

        fx = 0.0
        fy = 0.0
        for k in near_walls(grid, x, y, _NEIGHBOR_DIST):
            px, py = wall_point(seg, k, x, y)
            rx = x - px
            ry = y - py
            d = math.sqrt(rx * rx + ry * ry)
            if d < _EPS or d >= _NEIGHBOR_DIST:
                continue
            nx = rx / d
            ny = ry / d
            cos_theta = nx * ex + ny * ey
            mag = a_wall * (w + (1.0 - w) * (1.0 - cos_theta) * 0.5) * math.exp(-d / b_range)
            fx += mag * nx
            fy += mag * ny
        out[i, WALL, 0] = fx
        out[i, WALL, 1] = fy


class JohanssonPlanner(ForcePlanner):
    PARAM_DEFAULTS: ClassVar[dict[str, ParamDist]] = {
        "relaxation_time": ParamDist(0.5, 0.0, clip_low=0.05),
        "agent_scale": ParamDist(0.11, 0.0, clip_low=0.0),
        "obstacle_scale": ParamDist(0.11, 0.0, clip_low=0.0),
        "force_distance": ParamDist(1.19, 0.0, clip_low=0.01),
        "stride_time": ParamDist(0.5, 0.0, clip_low=0.0),
        "fov_weight": ParamDist(0.16, 0.0, clip_low=0.0, clip_high=1.0),
    }

    _kernel = staticmethod(_johansson_kernel)
