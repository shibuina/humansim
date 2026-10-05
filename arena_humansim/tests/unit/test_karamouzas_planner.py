from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.force import SOCIAL, WALL
from arena_humansim.local_planner.karamouzas import KaramouzasPlanner
from arena_humansim.utils.types import BeliefState, Pose2D

_DT = 0.05

Placement = tuple[float, float, float, float]


def _scene(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    planner: KaramouzasPlanner,
    others: Sequence[Placement],
) -> AgentPool:
    pool = pool_empty(capacity=len(others) + 1)
    planner.attach(pool)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    for k, (x, y, vx, vy) in enumerate(others):
        pool.add_agent(agent_factory(agent_id=k + 2, x=x, y=y))
        pool.vel[k + 1] = (vx, vy)
    pool.store_prev_vel()
    pool.set_goals({1: Pose2D(x=10.0, y=0.0)})
    return pool


def _social_of_first(planner: KaramouzasPlanner, pool: AgentPool, seen: Sequence[int]) -> np.ndarray:
    indptr = np.full(pool.n + 1, len(seen), dtype=np.int32)
    indptr[0] = 0
    pool.set_neighbor_csr(indptr, np.array(seen, dtype=np.int32))
    planner.compute_pool(pool, store_forces=True, dt=_DT)
    assert planner._last_force_arrays is not None
    return planner._last_force_arrays[2][0, SOCIAL].copy()


def _crowd(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], planner: KaramouzasPlanner) -> AgentPool:
    spots = [(0.0, 0.0), (1.1, 0.2), (2.0, -0.4), (0.4, 1.3), (-1.2, 0.6), (3.1, 0.9), (1.6, -1.5), (-0.8, -1.1)]
    pool = pool_empty(capacity=len(spots))
    planner.attach(pool)
    for k, (x, y) in enumerate(spots):
        pool.add_agent(agent_factory(agent_id=k + 1, x=x, y=y))
        pool.vel[k] = (math.cos(k), math.sin(2 * k))
        pool.theta[k] = 0.7 * k
    pool.store_prev_vel()
    pool.set_goals({k + 1: Pose2D(x=-x, y=4.0 - y) for k, (x, y) in enumerate(spots)})
    n = len(spots)
    indices = [j for i in range(n) for j in range(n) if j != i]
    pool.set_neighbor_csr(np.arange(0, n * (n - 1) + 1, n - 1, dtype=np.int32), np.array(indices, dtype=np.int32))
    return pool


def test_pooled_and_per_agent_paths_agree(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    planner.set_walls([((-5.0, -0.9), (5.0, -0.9))])
    agents = [agent_factory(agent_id=1, x=0.0, y=0.0), agent_factory(agent_id=2, x=2.5, y=0.3), agent_factory(agent_id=3, x=0.6, y=0.5)]
    agents[1].state.velocity = (-0.9, 0.1)
    agents[2].state.velocity = (0.3, -0.2)
    for agent in agents:
        agent.belief = BeliefState(observed_agents=[o.state for o in agents if o is not agent])
    goals = {1: Pose2D(x=10.0, y=0.0), 2: Pose2D(x=-10.0, y=0.0), 3: Pose2D(x=0.6, y=8.0)}

    pool = pool_empty(capacity=4)
    planner.attach(pool)
    for agent in agents:
        pool.add_agent(agent)
    pool.set_goals(goals)
    pool.set_neighbor_csr(np.array([0, 2, 4, 6], dtype=np.int32), np.array([1, 2, 0, 2, 0, 1], dtype=np.int32))
    planner.compute_pool(pool, dt=_DT)

    per_agent = planner.compute(agents, goals, dt=_DT)
    for i, agent in enumerate(agents):
        assert np.allclose(per_agent[agent.state.agent_id], pool.vel[i])


def test_single_head_on_neighbor_matches_piecewise_magnitude(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    pool = _scene(pool_empty, agent_factory, planner, [(3.0, 0.0, -1.0, 0.0)])
    pool.max_acceleration[0] = 1e3
    social = _social_of_first(planner, pool, [1])

    tau = agent_factory(agent_id=9).params.local_planner_params["relaxation_time"]
    des = pool.desired_vel[0] / tau * _DT
    circ = 1.0 + pool.agent_radius[1]
    tc = (3.0 - circ) / (des + 1.0)
    gap = (3.0 - tc) - des * tc
    d = des * tc + max(gap - pool.agent_radius[0] - pool.agent_radius[1], 0.0)
    assert d < 1.0
    assert np.allclose(social, (-3.0 / d, 0.0))


def test_only_most_imminent_neighbors_contribute(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    others = [(2.0, 0.3, -1.5, 0.0), (2.5, -0.3, -1.5, 0.0), (3.0, 0.2, -1.5, 0.0), (3.5, -0.2, -1.5, 0.0), (4.0, 0.1, -1.5, 0.0), (4.6, 0.0, -1.5, 0.0)]
    pool = _scene(pool_empty, agent_factory, planner, others)
    pool.max_acceleration[0] = 1e3

    least_imminent_alone = _social_of_first(planner, pool, [6])
    assert np.linalg.norm(least_imminent_alone) > 0.1
    top_five = _social_of_first(planner, pool, [3, 1, 5, 2, 4])
    all_six = _social_of_first(planner, pool, [6, 3, 1, 5, 2, 4])
    assert np.allclose(all_six, top_five)
    without_nearest = _social_of_first(planner, pool, [6, 3, 5, 2, 4])
    assert not np.allclose(without_nearest, _social_of_first(planner, pool, [3, 5, 2, 4]))


def test_colliding_neighbor_overrides_predictive_neighbors(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    pool = _scene(pool_empty, agent_factory, planner, [(3.0, 0.0, -1.5, 0.0), (-1.0, 0.0, 0.0, 0.0)])
    pool.max_acceleration[0] = 1e3

    predictive = _social_of_first(planner, pool, [1])
    colliding = _social_of_first(planner, pool, [2])
    both = _social_of_first(planner, pool, [1, 2])
    assert predictive[0] < 0.0
    assert colliding[0] > 0.0
    assert np.allclose(both, colliding)


def test_neighbors_outside_fov_are_ignored(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    pool = _scene(pool_empty, agent_factory, planner, [(-3.0, 0.0, 2.0, 0.0)])
    pool.max_acceleration[0] = 1e3

    assert np.allclose(_social_of_first(planner, pool, [1]), 0.0)
    pool.theta[0] = math.pi
    assert _social_of_first(planner, pool, [1])[0] > 0.0


def test_agent_force_is_capped_at_max_acceleration(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    pool = _scene(pool_empty, agent_factory, planner, [(1.0, 0.0, 0.0, 0.0)])

    social = _social_of_first(planner, pool, [1])
    assert np.isclose(np.linalg.norm(social), pool.max_acceleration[0])
    assert social[0] < 0.0
    pool.max_acceleration[0] = 0.7
    assert np.isclose(np.linalg.norm(_social_of_first(planner, pool, [1])), 0.7)


def test_wall_force_vanishes_beyond_wall_distance_and_grows_closer(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    planner.set_walls([((-50.0, 0.0), (50.0, 0.0))])
    pool = pool_empty(capacity=4)
    planner.attach(pool)
    for k, (x, y) in enumerate([(-20.0, 2.3), (0.0, 1.5), (20.0, 0.8)]):
        pool.add_agent(agent_factory(agent_id=k + 1, x=x, y=y))
    pool.set_goals({k + 1: Pose2D(x=40.0, y=y) for k, y in enumerate([2.3, 1.5, 0.8])})
    pool.set_neighbor_csr(np.zeros(4, dtype=np.int32), np.empty(0, dtype=np.int32))
    planner.compute_pool(pool, store_forces=True, dt=_DT)
    assert planner._last_force_arrays is not None
    wall = planner._last_force_arrays[2][:, WALL]
    assert np.allclose(wall[0], 0.0)
    assert 0.0 < wall[1, 1] < wall[2, 1]
    assert np.allclose(wall[:, 0], 0.0)


def test_wall_endpoint_repels(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    planner.set_walls([((0.0, 0.0), (1.0, 0.0))])
    pool = pool_empty(capacity=2)
    planner.attach(pool)
    pool.add_agent(agent_factory(agent_id=1, x=1.8, y=0.0))
    pool.set_goals({1: Pose2D(x=1.8, y=10.0)})
    pool.set_neighbor_csr(np.zeros(2, dtype=np.int32), np.empty(0, dtype=np.int32))
    planner.compute_pool(pool, store_forces=True, dt=_DT)
    assert planner._last_force_arrays is not None
    r = pool.agent_radius[0]
    safe = 2.0 + r
    assert np.allclose(planner._last_force_arrays[2][0, WALL], ((safe - 0.8) / (0.8 - r) ** 2, 0.0))


def test_two_runs_are_identical(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    walls = [((-3.0, -2.0), (4.0, -2.0)), ((4.0, -2.0), (4.0, 3.0))]
    runs = []
    for _ in range(2):
        planner = KaramouzasPlanner()
        planner.set_walls(walls)
        pool = _crowd(pool_empty, agent_factory, planner)
        planner.compute_pool(pool, dt=_DT)
        runs.append(pool.vel[: pool.n].copy())
    assert np.all(np.isfinite(runs[0]))
    assert np.array_equal(runs[0], runs[1])


def test_coincident_agents_stay_finite(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    pool = _scene(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.5, 0.0)])
    pool.set_goals({1: Pose2D(x=10.0, y=0.0), 2: Pose2D(x=-10.0, y=0.0)})
    pool.set_neighbor_csr(np.array([0, 1, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32))
    planner.compute_pool(pool, store_forces=True, dt=_DT)
    assert planner._last_force_arrays is not None
    assert np.all(np.isfinite(planner._last_force_arrays[2]))
    assert np.all(np.isfinite(pool.vel[:2]))
