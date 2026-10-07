from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.sfm import SFMPlanner
from arena_humansim.utils.types import Pose2D, Segments

WALL: Segments = [((0.0, 0.0), (5.0, 0.0))]


def _wall_force(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], offset: float) -> np.ndarray:
    pool = pool_empty(capacity=8)
    planner = SFMPlanner(wall_repulsion_range=0.05)
    planner.attach(pool)
    planner.set_walls(WALL)
    pool.add_agent(agent_factory(agent_id=1, x=-2.0, y=0.1))
    pool.theta[0] = 0.0
    pool.axial_offset[0] = offset
    pool.set_neighbor_csr(np.array([0, 0], dtype=np.int32), np.empty(0, dtype=np.int32))
    planner.compute_pool(pool, store_forces=True, dt=0.05)
    assert planner._last_force_arrays is not None
    return planner._last_force_arrays[4][0]


def test_capsule_far_disk_feels_wall_end_while_disk_at_same_center_feels_none(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    disk = _wall_force(pool_empty, agent_factory, 0.0)
    capsule = _wall_force(pool_empty, agent_factory, 1.8)
    assert np.array_equal(disk, np.zeros(2))
    assert capsule[0] < -1.0
    assert capsule[1] > 0.0


def _repulsion_on_first(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], offset: float) -> np.ndarray:
    pool = pool_empty(capacity=8)
    planner = SFMPlanner()
    planner.attach(pool)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.add_agent(agent_factory(agent_id=2, x=2.5, y=0.0))
    pool.theta[0] = 0.0
    pool.theta[1] = math.pi
    pool.axial_offset[:2] = offset
    pool.set_goals({1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=-5.0, y=0.0)})
    pool.set_neighbor_csr(np.array([0, 1, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32))
    planner.compute_pool(pool, store_forces=True, dt=0.05)
    assert planner._last_force_arrays is not None
    return planner._last_force_arrays[3][0]


def test_overlapping_capsule_ends_repel_while_disk_centers_do_not(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    disks = _repulsion_on_first(pool_empty, agent_factory, 0.0)
    capsules = _repulsion_on_first(pool_empty, agent_factory, 1.0)
    assert abs(disks[0]) < 0.01
    assert capsules[0] < -1.0
