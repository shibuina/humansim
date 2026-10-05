from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.force import SOCIAL, WALL
from arena_humansim.local_planner.zanlungo import ZanlungoPlanner, _wall_ttc
from arena_humansim.utils.types import BeliefState, Pose2D

_DT = 0.05


def _scene(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    planner: ZanlungoPlanner,
    agents: list[tuple[float, float, float, float]],
    neighbors: list[list[int]],
) -> np.ndarray:
    pool = pool_empty(capacity=max(4, len(agents)))
    planner.attach(pool)
    for k, (x, y, vx, vy) in enumerate(agents):
        pool.add_agent(agent_factory(agent_id=k + 1, x=x, y=y))
        pool.vel[k] = (vx, vy)
    pool.store_prev_vel()
    pool.set_goals({k + 1: Pose2D(x=x + 10.0 * vx, y=y + 10.0 * vy + 1.0) for k, (x, y, vx, vy) in enumerate(agents)})
    indptr = np.cumsum([0] + [len(nb) for nb in neighbors]).astype(np.int32)
    indices = np.array([j for nb in neighbors for j in nb], dtype=np.int32)
    pool.set_neighbor_csr(indptr, indices)
    planner.compute_pool(pool, store_forces=True, dt=_DT)
    assert planner._last_force_arrays is not None
    out = planner._last_force_arrays[2]
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(pool.vel[: pool.n]))
    return out


def test_pooled_and_per_agent_paths_agree(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    planner.set_walls([((-5.0, -0.8), (5.0, -0.8))])
    agents = [agent_factory(agent_id=1, x=0.0, y=0.0), agent_factory(agent_id=2, x=1.5, y=0.1)]
    agents[0].state.velocity = (0.8, -0.3)
    agents[1].state.velocity = (-0.6, 0.0)
    for agent, other in zip(agents, reversed(agents), strict=True):
        agent.belief = BeliefState(observed_agents=[other.state])
    goals = {1: Pose2D(x=10.0, y=0.0), 2: Pose2D(x=-10.0, y=0.0)}

    pool = pool_empty(capacity=4)
    planner.attach(pool)
    for agent in agents:
        pool.add_agent(agent)
    pool.set_goals(goals)
    pool.set_neighbor_csr(np.array([0, 1, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32))
    planner.compute_pool(pool, dt=_DT)

    per_agent = planner.compute(agents, goals, dt=_DT)
    for i, agent in enumerate(agents):
        assert np.allclose(per_agent[agent.state.agent_id], pool.vel[i])


def test_head_on_pair_repels_backward_and_laterally_before_contact(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    out = _scene(pool_empty, agent_factory, ZanlungoPlanner(), [(0.0, 0.0, 1.0, 0.0), (2.0, 0.1, -1.0, 0.0)], [[1], [0]])
    social = out[:, SOCIAL]
    assert social[0, 0] < 0.0
    assert social[0, 1] < 0.0
    assert social[1, 0] > 0.0
    assert social[1, 1] > 0.0
    assert np.allclose(social[0], -social[1])


def test_diverging_neighbor_gets_zero_force_while_wall_interaction_is_active(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    planner.set_walls([((-5.0, 2.0), (5.0, 2.0))])
    out = _scene(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 1.0), (0.0, -0.8, 0.0, -1.0)], [[1], []])
    assert np.all(out[0, SOCIAL] == 0.0)
    assert out[0, WALL, 1] < 0.0


def test_collision_time_takes_priority_over_earlier_close_approach(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    p = planner._defaults
    mass, a, b = p[1], p[2], p[4]
    mover = (0.0, 0.0, 1.0, 0.0)
    colliding = (4.0, 0.0, 0.0, 0.0)
    passing = (1.0, 0.7, 0.0, 0.0)

    approach_only = _scene(pool_empty, agent_factory, planner, [mover, passing], [[1], []])[0, SOCIAL]
    t_approach = 1.0
    expected = a * 1.0 / t_approach * math.exp(-(0.7 - 0.5) / b) / mass
    assert np.allclose(approach_only, (0.0, -expected))

    both = _scene(pool_empty, agent_factory, planner, [mover, colliding, passing], [[1, 2], [], []])[0, SOCIAL]
    t_collision = 3.5
    assert np.allclose(both, (-a * 1.0 / t_collision / mass, 0.0))


def test_isolated_agent_has_zero_social_and_wall_force(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    out = _scene(pool_empty, agent_factory, ZanlungoPlanner(), [(0.0, 0.0, 1.0, 0.3)], [[]])
    assert np.all(out[0, SOCIAL] == 0.0)
    assert np.all(out[0, WALL] == 0.0)


def test_wall_time_to_collision_from_both_sides_and_both_orientations() -> None:
    for seg in (np.array([[-5.0, 0.0, 5.0, 0.0]]), np.array([[5.0, 0.0, -5.0, 0.0]])):
        assert math.isclose(_wall_ttc(seg, 0, 0.0, 1.0, 0.0, -1.0, 0.25), 0.75)
        assert math.isclose(_wall_ttc(seg, 0, 0.0, -1.0, 0.0, 2.0, 0.25), 0.375)
        assert _wall_ttc(seg, 0, 0.0, 1.0, 0.0, 1.0, 0.25) == math.inf
        assert _wall_ttc(seg, 0, 0.0, -1.0, 0.0, -1.0, 0.25) == math.inf
        assert _wall_ttc(seg, 0, 1.0, 0.1, 0.0, 1.0, 0.25) == 0.0
        assert _wall_ttc(seg, 0, 1.0, 1.0, 0.0, 0.0, 0.25) == math.inf


def test_wall_time_to_collision_with_endpoint_cap_scales_with_speed() -> None:
    seg = np.array([[-5.0, 0.0, 5.0, 0.0]])
    gap = 1.0 - math.sqrt(0.25**2 - 0.1**2)
    assert math.isclose(_wall_ttc(seg, 0, -6.0, 0.1, 1.0, 0.0, 0.25), gap)
    assert math.isclose(_wall_ttc(seg, 0, -6.0, 0.1, 2.0, 0.0, 0.25), gap / 2.0)
    assert math.isclose(_wall_ttc(seg, 0, 6.0, -0.1, -2.0, 0.0, 0.25), gap / 2.0)
    assert _wall_ttc(seg, 0, -6.0, 0.4, 1.0, 0.0, 0.25) == math.inf


def test_wall_force_points_away_from_wall_on_either_side(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    planner.set_walls([((-5.0, 0.0), (5.0, 0.0))])
    p = planner._defaults
    mass, a_wall = p[1], p[3]
    out = _scene(pool_empty, agent_factory, planner, [(0.0, 1.0, 0.0, -1.0), (2.0, -1.0, 0.0, 1.0)], [[], []])
    wall = out[:, WALL]
    expected = a_wall * 1.0 / 0.75 / mass
    assert np.allclose(wall[0], (0.0, expected))
    assert np.allclose(wall[1], (0.0, -expected))


def test_wall_force_keeps_agent_side_when_prediction_crosses_wall(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    planner.set_walls([((-5.0, 0.0), (5.0, 0.0))])
    out = _scene(pool_empty, agent_factory, planner, [(0.0, 1.0, 0.0, -1.0), (0.0, -1.0, 0.0, 0.0)], [[1], []])
    assert out[0, WALL, 1] > 0.0
    assert out[0, WALL, 0] == 0.0


def test_coincident_and_resting_agents_stay_finite(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    planner.set_walls([((-5.0, 0.0), (5.0, 0.0)), ((1.0, 1.0), (1.0, 1.0))])
    scenes = (
        [(0.0, 0.5, 0.0, 0.0), (0.0, 0.5, 0.0, 0.0)],
        [(0.0, 0.5, 0.5, 0.0), (0.0, 0.5, 0.5, 0.0)],
        [(0.0, 0.5, 0.5, 0.0), (0.0, 0.5, -0.5, 0.2)],
        [(1.0, 1.0, 0.3, 0.3), (0.0, 0.0, 0.0, 0.0)],
    )
    for agents in scenes:
        _scene(pool_empty, agent_factory, planner, agents, [[1], [0]])
