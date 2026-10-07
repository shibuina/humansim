from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.collision.wall_projection import WallProjectionResolver
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool

_MARGIN = 0.01
_DOORWAY_WALLS = [((0.0, -5.0), (0.0, -0.45)), ((0.0, 0.45), (0.0, 5.0))]
_DOORWAY_RADIUS = 0.325
_DOORWAY_LENGTH = 1.1
_DT = 0.05


def _capsule_pool(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], rows: list[tuple[float, float, float, float, float]]) -> AgentPool:
    pool = pool_empty(capacity=8)
    for i, (x, y, theta, radius, offset) in enumerate(rows):
        pool.add_agent(agent_factory(i + 1, x=x, y=y))
        pool.theta[i] = theta
        pool.agent_radius[i] = radius
        pool.axial_offset[i] = offset
        pool.policy_idx[i] = 0
    return pool


def test_capsule_end_crossing_wall_is_pushed_out_while_disk_center_is_clear(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls([((-5.0, 1.0), (5.0, 1.0))])
    pool = _capsule_pool(pool_empty, agent_factory, [(0.0, 0.65, math.pi / 2, 0.25, 0.3)])
    pool.vel[0] = (0.5, 1.0)

    corrected = resolver.resolve(pool)

    assert corrected == {1}
    assert pool.pos[0] == pytest.approx((0.0, 0.44), abs=1e-12)
    assert pool.vel[0] == pytest.approx((0.5, 0.0), abs=1e-12)
    assert pool.theta[0] == math.pi / 2


def test_disk_at_same_center_is_untouched(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls([((-5.0, 1.0), (5.0, 1.0))])
    pool = _capsule_pool(pool_empty, agent_factory, [(0.0, 0.65, math.pi / 2, 0.25, 0.0)])
    pool.vel[0] = (0.5, 1.0)

    corrected = resolver.resolve(pool)

    assert corrected == set()
    assert pool.pos[0].tolist() == [0.0, 0.65]
    assert pool.vel[0].tolist() == [0.5, 1.0]


def test_capsules_with_overlapping_ends_separate_and_stop_closing(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _capsule_pool(pool_empty, agent_factory, [(0.0, 0.0, 0.0, 0.25, 0.5), (1.3, 0.0, 0.0, 0.25, 0.5)])
    pool.vel[0] = (1.0, 0.0)
    pool.vel[1] = (-1.0, 0.0)

    WallProjectionResolver(margin=_MARGIN).resolve(pool)

    assert pool.pos[0] == pytest.approx((-0.1, 0.0), abs=1e-12)
    assert pool.pos[1] == pytest.approx((1.4, 0.0), abs=1e-12)
    assert pool.vel[0] == pytest.approx((0.0, 0.0), abs=1e-12)
    assert pool.vel[1] == pytest.approx((0.0, 0.0), abs=1e-12)


def test_disks_at_same_centers_do_not_touch(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _capsule_pool(pool_empty, agent_factory, [(0.0, 0.0, 0.0, 0.25, 0.0), (1.3, 0.0, 0.0, 0.25, 0.0)])
    pool.vel[0] = (1.0, 0.0)
    pool.vel[1] = (-1.0, 0.0)

    WallProjectionResolver(margin=_MARGIN).resolve(pool)

    assert pool.pos[:2].tolist() == [[0.0, 0.0], [1.3, 0.0]]
    assert pool.vel[:2].tolist() == [[1.0, 0.0], [-1.0, 0.0]]


def test_capsule_end_beyond_disk_query_radius_is_still_separated(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _capsule_pool(pool_empty, agent_factory, [(0.0, 0.0, 0.0, 0.25, 1.0), (2.3, 0.0, 0.0, 0.25, 1.0)])

    WallProjectionResolver(margin=_MARGIN).resolve(pool)

    assert pool.pos[1, 0] - pool.pos[0, 0] == pytest.approx(2.5, abs=1e-12)


def _drive_through_doorway(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], theta: float) -> float:
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls(_DOORWAY_WALLS)
    pool = _capsule_pool(pool_empty, agent_factory, [(-2.0, 0.0, theta, _DOORWAY_RADIUS, _DOORWAY_LENGTH / 2 - _DOORWAY_RADIUS)])
    for _ in range(60):
        pool.vel[0] = (1.0, 0.0)
        pool.pos[0] += pool.vel[0] * _DT
        resolver.resolve(pool)
    return float(pool.pos[0, 0])


def test_capsule_aligned_with_doorway_passes_through(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    assert _drive_through_doorway(pool_empty, agent_factory, 0.0) == pytest.approx(1.0, abs=1e-9)


def test_capsule_perpendicular_to_doorway_is_held_back(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    held_x = -math.sqrt((_DOORWAY_RADIUS + _MARGIN) ** 2 - (0.45 - (_DOORWAY_LENGTH / 2 - _DOORWAY_RADIUS)) ** 2)
    final_x = _drive_through_doorway(pool_empty, agent_factory, math.pi / 2)
    assert held_x - 0.001 < final_x <= held_x
    assert final_x == pytest.approx(-0.2485, abs=5e-5)


def test_rows_beyond_n_and_orientation_untouched_when_a_capsule_is_resolved(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls([((-5.0, 1.0), (5.0, 1.0))])
    pool = _capsule_pool(pool_empty, agent_factory, [(0.0, 0.65, math.pi / 2, 0.25, 0.3), (0.7, 0.65, 0.3, 0.25, 0.0)])
    pool.pos[2:] = np.arange(12, dtype=np.float64).reshape(6, 2) * 0.37
    pool.vel[2:] = np.arange(12, dtype=np.float64).reshape(6, 2) * -0.11
    pool.theta[2:] = np.arange(6, dtype=np.float64) * 0.5
    pos_before = pool.pos.copy()
    vel_before = pool.vel.copy()
    theta_before = pool.theta.copy()

    corrected = resolver.resolve(pool)

    assert corrected == {1}
    assert np.array_equal(pool.pos[2:], pos_before[2:])
    assert np.array_equal(pool.vel[2:], vel_before[2:])
    assert np.array_equal(pool.theta, theta_before)
