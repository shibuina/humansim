"""Karamouzas, Heil, van Beek, Overmars 2009 predictive collision avoidance, after Menge's AgtKaramouzas."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from numba import njit, prange

from arena_humansim.core.agents.types import ParamDist

from .force import GOAL, SOCIAL, WALL, ForcePlanner, WallGridArgs, near_walls, ray_circle_ttc, wall_point

_EPS = 1e-6
_MENGE_EPS = 0.01
_WEIGHT_DECAY = 0.8

_TAU, _STEEPNESS, _WALL_DIST, _COUNT, _D_MIN, _D_MID, _D_MAX, _AGENT_FORCE, _PERSONAL, _ANTICIPATION, _FOV = range(11)


@njit(cache=True)
def _magnitude(d: float, d_min: float, d_mid: float, d_max: float, agent_force: float) -> float:
    if d < d_min:
        return agent_force * d_min / d
    if d < d_mid:
        return agent_force
    if d < d_max:
        return agent_force * (d_max - d) / (d_max - d_mid)
    return 0.0


@njit(cache=True, parallel=True)
def _karamouzas_kernel(
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
    n = pos.shape[0]
    k_max = 1
    for i in range(n):
        k_max = max(k_max, int(params[i, _COUNT]))
    buf_t = np.empty((n, k_max), dtype=np.float64)
    buf_j = np.empty((n, k_max), dtype=np.int64)

    for i in prange(n):
        x = pos[i, 0]
        y = pos[i, 1]
        vx = vel[i, 0]
        vy = vel[i, 1]
        r = radius[i]
        tau = params[i, _TAU]
        steepness = params[i, _STEEPNESS]
        k = int(params[i, _COUNT])
        d_min = params[i, _D_MIN]
        d_mid = params[i, _D_MID]
        d_max = params[i, _D_MAX]
        agent_force = params[i, _AGENT_FORCE]
        personal = params[i, _PERSONAL]
        anticipation = params[i, _ANTICIPATION]
        cos_fov = math.cos(0.5 * math.radians(params[i, _FOV]))

        gx = (pref[i, 0] - vx) / tau
        gy = (pref[i, 1] - vy) / tau
        out[i, GOAL, 0] = gx
        out[i, GOAL, 1] = gy

        safe = params[i, _WALL_DIST] + r
        wx = 0.0
        wy = 0.0
        for w in near_walls(grid, x, y, safe):
            px, py = wall_point(seg, w, x, y)
            ox = x - px
            oy = y - py
            d_sq = ox * ox + oy * oy
            if d_sq >= safe * safe:
                continue
            d = math.sqrt(d_sq)
            if d < _EPS:
                continue
            mag = (safe - d) / math.pow(max(d - r, _MENGE_EPS), steepness)
            wx += ox / d * mag
            wy += oy / d * mag
        out[i, WALL, 0] = wx
        out[i, WALL, 1] = wy

        dvx = vx + (gx + wx) * dt
        dvy = vy + (gy + wy) * dt
        des_speed = math.hypot(dvx, dvy)
        orient_x = math.cos(theta[i])
        orient_y = math.sin(theta[i])

        colliding = False
        cfx = 0.0
        cfy = 0.0
        m = 0
        for t in range(indptr[i], indptr[i + 1]):
            j = indices[t]
            rx = pos[j, 0] - x
            ry = pos[j, 1] - y
            circ = personal + radius[j]
            d_sq = rx * rx + ry * ry
            if d_sq < circ * circ:
                colliding = True
                d = math.sqrt(d_sq)
                if d < _EPS:
                    continue
                mag = _magnitude(max(max(d - r - radius[j], 0.0), _MENGE_EPS), d_min, d_mid, d_max, agent_force)
                cfx -= rx / d * mag
                cfy -= ry / d * mag
                continue
            if colliding:
                continue
            d = math.sqrt(d_sq)
            if d < _EPS or (rx * orient_x + ry * orient_y) / d < cos_fov:
                continue
            tc = ray_circle_ttc(dvx - vel[j, 0], dvy - vel[j, 1], rx, ry, circ)
            if not tc < anticipation:
                continue
            p = 0
            while p < m and tc > buf_t[i, p]:
                p += 1
            if p >= k:
                continue
            for q in range(min(m, k - 1), p, -1):
                buf_t[i, q] = buf_t[i, q - 1]
                buf_j[i, q] = buf_j[i, q - 1]
            buf_t[i, p] = tc
            buf_j[i, p] = j
            m = min(m + 1, k)

        if colliding:
            fx = cfx
            fy = cfy
        else:
            fx = 0.0
            fy = 0.0
            weight = 1.0
            for q in range(m):
                j = buf_j[i, q]
                tc = buf_t[i, q]
                ex = x + dvx * tc - pos[j, 0] - vel[j, 0] * tc
                ey = y + dvy * tc - pos[j, 1] - vel[j, 1] * tc
                d = math.hypot(ex, ey)
                if d < _EPS:
                    continue
                gap = max(des_speed * tc + max(d - r - radius[j], 0.0), _MENGE_EPS)
                if gap >= d_max:
                    continue
                mag = _magnitude(gap, d_min, d_mid, d_max, agent_force) * weight
                weight *= _WEIGHT_DECAY
                fx += ex / d * mag
                fy += ey / d * mag

        f = math.hypot(fx, fy)
        if f > max_acc[i]:
            fx *= max_acc[i] / f
            fy *= max_acc[i] / f
        out[i, SOCIAL, 0] = fx
        out[i, SOCIAL, 1] = fy


class KaramouzasPlanner(ForcePlanner):
    PARAM_DEFAULTS: ClassVar[dict[str, ParamDist]] = {
        "relaxation_time": ParamDist(0.4, 0.0, clip_low=0.05),
        "wall_steepness": ParamDist(2.0, 0.0, clip_low=0.0),
        "wall_distance": ParamDist(2.0, 0.0, clip_low=0.0),
        "colliding_count": ParamDist(5.0, 0.0, clip_low=1.0),
        "d_min": ParamDist(1.0, 0.0, clip_low=0.01),
        "d_mid": ParamDist(8.0, 0.0, clip_low=0.01),
        "d_max": ParamDist(10.0, 0.0, clip_low=0.01),
        "agent_force": ParamDist(3.0, 0.0, clip_low=0.0),
        "personal_space": ParamDist(1.0, 0.0, clip_low=0.0),
        "anticipation": ParamDist(3.0, 0.0, clip_low=0.0),
        "fov_angle": ParamDist(200.0, 0.0, clip_low=0.0, clip_high=360.0),
    }

    _kernel = staticmethod(_karamouzas_kernel)
