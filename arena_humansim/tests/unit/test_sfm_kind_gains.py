from __future__ import annotations

from collections.abc import Callable

import numpy as np

from arena_humansim.core import interaction_classes
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.local_planner import LocalPlanner
from arena_humansim.local_planner.sfm import SFMPlanner
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.types import Pose2D


def _build_pool_and_planner(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    kinds: list[int],
) -> tuple[AgentPool, SFMPlanner]:
    pool = pool_empty(capacity=8)
    planner = SFMPlanner()
    planner.attach(pool)
    for i, k in enumerate(kinds):
        pool.add_agent(agent_factory(agent_id=i + 1, x=float(i) * 0.5, y=0.0))
        pool.kind[i] = min(k, 1)
        pool.interaction_class[i] = k
        pool.policy_idx[i] = 0
    pool.set_goals({1: Pose2D(x=10.0, y=0.0)})
    indptr = np.array([0, 1, 1], dtype=np.int32)
    indices = np.array([1], dtype=np.int32)
    pool.set_neighbor_csr(indptr, indices)
    return pool, planner


def test_robot_neighbor_produces_larger_repulsion(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    baseline, p1 = _build_pool_and_planner(pool_empty, agent_factory, [0, 0])
    p1.compute_pool(baseline, store_forces=False, dt=0.05)
    baseline_vel = baseline.vel[0].copy()

    robot, p2 = _build_pool_and_planner(pool_empty, agent_factory, [0, 1])
    p2.compute_pool(robot, store_forces=False, dt=0.05)
    robot_vel = robot.vel[0].copy()

    assert robot_vel[0] < baseline_vel[0]


def test_unknown_third_class_gets_unit_gains(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    third = interaction_classes.index("golf_cart")
    baseline, p1 = _build_pool_and_planner(pool_empty, agent_factory, [0, 0])
    p1.compute_pool(baseline, store_forces=False, dt=0.05)

    other, p2 = _build_pool_and_planner(pool_empty, agent_factory, [0, third])
    p2.compute_pool(other, store_forces=False, dt=0.05)

    assert p2._gain_strength_scale.shape[0] > third
    assert p2._gain_strength_scale[0, third] == 1.0
    assert p2._gain_range_scale[0, third] == 1.0
    assert np.array_equal(other.vel[0], baseline.vel[0])


def test_wheelchair_default_gains_land_once_registered(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    wheelchair = interaction_classes.index("wheelchair")
    baseline, p1 = _build_pool_and_planner(pool_empty, agent_factory, [0, 0])
    p1.compute_pool(baseline, store_forces=False, dt=0.05)

    pool, p2 = _build_pool_and_planner(pool_empty, agent_factory, [0, wheelchair])
    p2.compute_pool(pool, store_forces=False, dt=0.05)

    assert p2._gain_strength_scale[0, wheelchair] == 1.3
    assert p2._gain_range_scale[0, wheelchair] == 1.4
    assert pool.vel[0, 0] < baseline.vel[0, 0]


def test_wheelchair_observer_keeps_the_human_distance_from_robots(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    wheelchair = interaction_classes.index("wheelchair")
    robot = interaction_classes.ROBOT
    human, p1 = _build_pool_and_planner(pool_empty, agent_factory, [0, robot])
    p1.compute_pool(human, store_forces=False, dt=0.05)
    peer, p2 = _build_pool_and_planner(pool_empty, agent_factory, [wheelchair, wheelchair])
    p2.compute_pool(peer, store_forces=False, dt=0.05)

    pool, p3 = _build_pool_and_planner(pool_empty, agent_factory, [wheelchair, robot])
    p3.compute_pool(pool, store_forces=False, dt=0.05)

    assert p3._gain_strength_scale[wheelchair, robot] == p3._gain_strength_scale[0, robot] == 1.5
    assert p3._gain_range_scale[wheelchair, robot] == p3._gain_range_scale[0, robot] == 1.3
    assert p3._gain_strength_scale[robot, wheelchair] == 1.0
    assert p2._gain_strength_scale[wheelchair, wheelchair] == 1.0
    assert pool.vel[0, 0] == human.vel[0, 0]
    assert pool.vel[0, 0] < peer.vel[0, 0]


def test_apply_policy_params_registers_new_class_name() -> None:
    p = SFMPlanner()
    p.apply_policy_params('{"kind_gains": {"human_scooter": {"strength_scale": 2.5, "range_scale": 1.1}}}')
    scooter = interaction_classes.index("scooter")
    assert p._gain_strength_scale[0, scooter] == 2.5
    assert p._gain_range_scale[0, scooter] == 1.1
    assert p._gain_strength_scale[0, 1] == 1.5
    assert p._gain_range_scale[0, 1] == 1.3


def test_apply_policy_params_parses_overrides() -> None:
    p = SFMPlanner()
    p.apply_policy_params('{"kind_gains": {"human_robot": {"strength_scale": 3.0, "range_scale": 2.0}}}')
    assert p._gain_strength_scale[0, 1] == 3.0
    assert p._gain_range_scale[0, 1] == 2.0


def test_apply_policy_params_defensive_against_garbage() -> None:
    p = SFMPlanner()
    before_s = p._gain_strength_scale.copy()
    before_r = p._gain_range_scale.copy()
    for bad in ["", "not json", "[1,2,3]", '{"kind_gains": "nope"}', '{"kind_gains": {"bad": {}}}']:
        p.apply_policy_params(bad)
    assert np.array_equal(p._gain_strength_scale, before_s)
    assert np.array_equal(p._gain_range_scale, before_r)


def test_registry_includes_straight() -> None:
    assert "straight" in LocalPlanner.list_available()
