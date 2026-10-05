from __future__ import annotations

from collections.abc import Callable

import numpy as np

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.force import GOAL, SOCIAL, WALL
from arena_humansim.local_planner.helbing import HelbingPlanner
from arena_humansim.utils.types import BeliefState, Pose2D


def _pair_pool(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    gap: float,
    planner: HelbingPlanner,
    vel_j: tuple[float, float] = (0.0, 0.0),
) -> AgentPool:
    pool = pool_empty(capacity=4)
    planner.attach(pool)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.add_agent(agent_factory(agent_id=2, x=gap, y=0.0))
    pool.vel[1] = vel_j
    pool.store_prev_vel()
    pool.set_goals({1: Pose2D(x=10.0, y=0.0), 2: Pose2D(x=-10.0, y=0.0)})
    pool.set_neighbor_csr(np.array([0, 1, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32))
    return pool


def _social(planner: HelbingPlanner, pool: AgentPool) -> np.ndarray:
    planner.compute_pool(pool, store_forces=True, dt=0.05)
    assert planner._last_force_arrays is not None
    return planner._last_force_arrays[2][:, SOCIAL]


def test_contact_terms_only_under_overlap(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    r_ij = 2 * agent_factory(agent_id=9).params.agent_radius
    p = planner._defaults
    a, b, mass, k = p[2], p[4], p[1], p[5]

    apart = _social(planner, _pair_pool(pool_empty, agent_factory, r_ij + 0.1, planner))
    expected_apart = a * np.exp((r_ij - (r_ij + 0.1)) / b) / mass
    assert np.isclose(-apart[0, 0], expected_apart)

    g = 0.02
    overlap = _social(planner, _pair_pool(pool_empty, agent_factory, r_ij - g, planner))
    expected_overlap = (a * np.exp(g / b) + k * g) / mass
    assert np.isclose(-overlap[0, 0], expected_overlap)
    assert np.isclose(overlap[0, 1], 0.0)


def test_sliding_friction_opposes_relative_tangential_velocity(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    r_ij = 2 * agent_factory(agent_id=9).params.agent_radius
    social = _social(planner, _pair_pool(pool_empty, agent_factory, r_ij - 0.02, planner, vel_j=(0.0, 0.5)))
    assert social[0, 1] > 0.0
    assert social[1, 1] < 0.0
    assert np.allclose(social[0], -social[1])


def test_wall_force_points_away_and_decays(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    planner.set_walls([((-5.0, -1.0), (5.0, -1.0))])
    pool = pool_empty(capacity=4)
    planner.attach(pool)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=-0.5))
    pool.add_agent(agent_factory(agent_id=2, x=3.0, y=-0.7))
    pool.set_goals({1: Pose2D(x=0.0, y=-0.5), 2: Pose2D(x=10.0, y=-0.7)})
    pool.set_neighbor_csr(np.zeros(3, dtype=np.int32), np.empty(0, dtype=np.int32))
    planner.compute_pool(pool, store_forces=True, dt=0.05)
    assert planner._last_force_arrays is not None
    wall = planner._last_force_arrays[2][:, WALL]
    assert wall[0, 1] > 0.0
    assert wall[1, 1] > wall[0, 1]
    assert np.allclose(wall[:, 0], 0.0)


def test_goal_term_relaxes_toward_preferred_velocity(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    pool = _pair_pool(pool_empty, agent_factory, 50.0, planner)
    pool.set_neighbor_csr(np.zeros(3, dtype=np.int32), np.empty(0, dtype=np.int32))
    planner.compute_pool(pool, store_forces=True, dt=0.05)
    assert planner._last_force_arrays is not None
    goal = planner._last_force_arrays[2][:, GOAL]
    tau = planner._defaults[0]
    assert np.allclose(goal[0], (pool.desired_vel[0] / tau, 0.0))
    assert np.allclose(goal[1], (-pool.desired_vel[1] / tau, 0.0))


def test_pooled_and_per_agent_paths_agree(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    planner.set_walls([((-5.0, -0.6), (5.0, -0.6))])
    agents = [agent_factory(agent_id=1, x=0.0, y=0.0), agent_factory(agent_id=2, x=0.45, y=0.1)]
    agents[1].state.velocity = (-0.4, 0.2)
    for agent, other in zip(agents, reversed(agents), strict=True):
        agent.belief = BeliefState(observed_agents=[other.state])
    goals = {1: Pose2D(x=10.0, y=0.0), 2: Pose2D(x=-10.0, y=0.0)}

    pool = pool_empty(capacity=4)
    planner.attach(pool)
    for agent in agents:
        pool.add_agent(agent)
    pool.set_goals(goals)
    pool.set_neighbor_csr(np.array([0, 1, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32))
    planner.compute_pool(pool, dt=0.05)

    per_agent = planner.compute(agents, goals, dt=0.05)
    for i, agent in enumerate(agents):
        assert np.allclose(per_agent[agent.state.agent_id], pool.vel[i])


def test_coincident_agents_stay_finite(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    pool = _pair_pool(pool_empty, agent_factory, 0.0, planner)
    planner.compute_pool(pool, dt=0.05)
    assert np.all(np.isfinite(pool.vel[:2]))
