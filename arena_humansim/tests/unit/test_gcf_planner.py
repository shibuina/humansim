from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.force import SOCIAL, WALL
from arena_humansim.local_planner.gcf import (
    GCFPlanner,
    closest_approach,
    distance_response,
    ellipse_axes,
    polar_radius,
)
from arena_humansim.utils.types import BeliefState, Pose2D

_NO_NEIGHBORS = (np.zeros(3, dtype=np.int32), np.empty(0, dtype=np.int32))
_PAIR = (np.array([0, 1, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32))


def _support_reference(a1: float, b1: float, th1: float, a2: float, b2: float, th2: float, psi: float) -> float:
    phi = np.linspace(-math.pi, math.pi, 40001)
    nx = np.cos(phi)
    ny = np.sin(phi)

    def support(a: float, b: float, th: float) -> np.ndarray:
        u = nx * math.cos(th) + ny * math.sin(th)
        v = ny * math.cos(th) - nx * math.sin(th)
        return np.sqrt(a * a * u * u + b * b * v * v)

    nd = nx * math.cos(psi) + ny * math.sin(psi)
    front = nd > 1e-9
    return float(np.min((support(a1, b1, th1) + support(a2, b2, th2))[front] / nd[front]))


def _pool(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    planner: GCFPlanner,
    states: list[tuple[float, float, float, float, float]],
    goals: list[tuple[float, float]],
    csr: tuple[np.ndarray, np.ndarray],
) -> AgentPool:
    pool = pool_empty(capacity=4)
    planner.attach(pool)
    for k, (x, y, theta, vx, vy) in enumerate(states):
        pool.add_agent(agent_factory(agent_id=k + 1, x=x, y=y))
        pool.theta[k] = theta
        pool.vel[k] = (vx, vy)
    pool.store_prev_vel()
    pool.set_goals({k + 1: Pose2D(x=gx, y=gy) for k, (gx, gy) in enumerate(goals)})
    pool.set_neighbor_csr(*csr)
    return pool


def _forces(planner: GCFPlanner, pool: AgentPool) -> np.ndarray:
    planner.compute_pool(pool, store_forces=True, dt=0.05)
    assert planner._last_force_arrays is not None
    return planner._last_force_arrays[2]


def test_closest_approach_matches_support_function_reference() -> None:
    rng = np.random.default_rng(7)
    axis_angles = (0.0, math.pi / 2, math.pi, -math.pi / 2)
    for k in range(300):
        a1, a2 = rng.uniform(0.15, 1.2, 2)
        b1, b2 = rng.uniform(0.15, 0.4, 2)
        if k % 5 == 0:
            b1 = a1 * (1.0 + rng.normal(0.0, 1e-4))
        th1 = float(rng.choice(axis_angles)) if k % 3 == 0 else rng.uniform(-math.pi, math.pi)
        th2 = th1 + float(rng.choice((0.0, math.pi))) + rng.normal(0.0, 1e-5) if k % 4 == 0 else rng.uniform(-math.pi, math.pi)
        psi = th1 if k % 7 == 0 else rng.uniform(-math.pi, math.pi)
        got = closest_approach(a1, b1, math.cos(th1), math.sin(th1), a2, b2, math.cos(th2), math.sin(th2), math.cos(psi), math.sin(psi))
        ref = _support_reference(a1, b1, th1, a2, b2, th2, psi)
        assert got == pytest.approx(ref, abs=2e-3), (k, a1, b1, th1, a2, b2, th2, psi)


def test_closest_approach_of_circles_is_sum_of_radii() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        r1, r2 = rng.uniform(0.1, 0.6, 2)
        th1, th2, psi = rng.uniform(-math.pi, math.pi, 3)
        got = closest_approach(r1, r1, math.cos(th1), math.sin(th1), r2, r2, math.cos(th2), math.sin(th2), math.cos(psi), math.sin(psi))
        assert got == pytest.approx(r1 + r2, abs=1e-9)


def test_polar_radius_lies_on_the_ellipse() -> None:
    a, b, th = 0.6, 0.25, 0.7
    for psi in np.linspace(-math.pi, math.pi, 37):
        r = polar_radius(a, b, math.cos(th), math.sin(th), math.cos(psi), math.sin(psi))
        u = r * math.cos(psi - th)
        v = r * math.sin(psi - th)
        assert (u / a) ** 2 + (v / b) ** 2 == pytest.approx(1.0)


def test_distance_response_is_continuous_and_truncated() -> None:
    max_dist, width, max_force = 2.0, 0.12, 3.0
    for x in (0.0, width, max_dist - width, max_dist):
        assert distance_response(x - 1e-9, max_dist, width, max_force) == pytest.approx(distance_response(x + 1e-9, max_dist, width, max_force), abs=1e-6)
    assert distance_response(-0.3, max_dist, width, max_force) == 3.0 * max_force
    assert distance_response(1.0, max_dist, width, max_force) == pytest.approx(1.0)
    for x in (max_dist, max_dist + 0.5, 10.0):
        assert distance_response(x, max_dist, width, max_force) == 0.0
    samples = [distance_response(x, max_dist, width, max_force) for x in np.linspace(width, max_dist, 400)]
    assert np.all(np.diff(samples) <= 1e-12)


def test_ellipse_axes_follow_speed() -> None:
    p = GCFPlanner.PARAM_DEFAULTS
    a_min, a_rate, b_max, b_growth = p["a_min"].mean, p["a_rate"].mean, p["b_max"].mean, p["b_growth"].mean
    assert ellipse_axes(a_min, a_rate, b_max, b_growth, 0.0) == pytest.approx((a_min, b_max))
    a, b = ellipse_axes(a_min, a_rate, b_max, b_growth, 1.3)
    assert a == pytest.approx(a_min + a_rate * 1.3)
    assert b == pytest.approx(b_max - b_growth)
    assert ellipse_axes(a_min, a_rate, b_max, b_growth, 100.0)[1] > 0.0


def test_faster_leader_elongates_and_pushes_harder(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    pushes = []
    for speed in (0.6, 0.9, 1.2):
        pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 0.6, 0.0), (1.4, 0.0, 0.0, speed, 0.0)], [(10.0, 0.0), (10.0, 0.0)], _PAIR)
        social = _forces(planner, pool)[:, SOCIAL]
        assert social[0, 0] < 0.0
        assert social[0, 1] == pytest.approx(0.0)
        pushes.append(-social[0, 0])
    assert pushes[0] < pushes[1] < pushes[2]


def test_converging_pair_follows_paper_velocity_term(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    gap, v_i, v_j = 1.5, 0.8, -0.4
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, v_i, 0.0), (gap, 0.0, math.pi, v_j, 0.0)], [(10.0, 0.0), (-10.0, 0.0)], _PAIR)
    social = _forces(planner, pool)[:, SOCIAL]
    p = planner._defaults
    nu, max_dist, max_force, width = p[1], p[2], p[3], p[4]
    a_i, b_i = ellipse_axes(p[5], p[6], p[7], p[8], abs(v_i))
    a_j, b_j = ellipse_axes(p[5], p[6], p[7], p[8], abs(v_j))
    dca = closest_approach(a_i, b_i, 1.0, 0.0, a_j, b_j, -1.0, 0.0, 1.0, 0.0)
    v0 = float(pool.desired_vel[0])
    expected = distance_response(gap - dca, max_dist, width, max_force) * (nu * v0 + (v_i - v_j)) ** 2
    assert -social[0, 0] == pytest.approx(expected)
    assert social[0, 1] == pytest.approx(0.0)


def test_neighbors_beside_behind_or_seen_while_standing_exert_no_force(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    for neighbor, velocity in (((-0.8, 0.0), (0.6, 0.0)), ((0.0, 0.8), (0.6, 0.0)), ((0.8, 0.0), (0.0, 0.0))):
        pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, *velocity), (*neighbor, 0.0, 0.0, 0.0)], [(10.0, 0.0), (10.0, 0.0)], _PAIR)
        assert np.allclose(_forces(planner, pool)[0, SOCIAL], 0.0)


def test_neighbor_weight_is_cosine_of_bearing_from_velocity(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    ahead = _forces(planner, _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 0.0, 0.0), (1.2, 0.0, 0.0, 0.0, 0.0)], [(10.0, 0.0), (10.0, 0.0)], _PAIR))
    assert np.allclose(ahead[0, SOCIAL], 0.0)
    magnitudes = []
    for bearing in (0.0, math.pi / 3):
        heading = (0.001 * math.cos(bearing), -0.001 * math.sin(bearing))
        pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, *heading), (1.2, 0.0, 0.0, 0.0, 0.0)], [(10.0, 0.0), (10.0, 0.0)], _PAIR)
        magnitudes.append(float(np.hypot(*_forces(planner, pool)[0, SOCIAL])))
    assert magnitudes[1] / magnitudes[0] == pytest.approx(0.5, rel=2e-2)


def test_walking_parallel_to_a_wall_or_standing_feels_no_wall_force(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    planner.set_walls([((-5.0, -0.5), (5.0, -0.5))])
    pool = _pool(
        pool_empty,
        agent_factory,
        planner,
        [(0.0, 0.0, 0.0, 0.8, 0.0), (2.0, 0.0, 0.0, 0.0, 0.0), (-2.0, 0.0, math.pi / 2, 0.0, 0.5)],
        [(10.0, 0.0), (10.0, 0.0), (-2.0, 10.0)],
        (np.zeros(4, dtype=np.int32), np.empty(0, dtype=np.int32)),
    )
    assert np.allclose(_forces(planner, pool)[:, WALL], 0.0)


def test_wall_force_points_away_and_decays(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    planner.set_walls([((-5.0, -1.0), (5.0, -1.0))])
    pool = _pool(
        pool_empty,
        agent_factory,
        planner,
        [(0.0, -0.5, -math.pi / 2, 0.0, -0.5), (0.0, 0.0, -math.pi / 2, 0.0, -0.5), (0.0, 0.8, -math.pi / 2, 0.0, -0.5)],
        [(0.0, -10.0), (0.0, -10.0), (0.0, -10.0)],
        (np.zeros(4, dtype=np.int32), np.empty(0, dtype=np.int32)),
    )
    wall = _forces(planner, pool)[:, WALL]
    assert np.all(wall[:, 1] > 0.0)
    assert wall[0, 1] > wall[1, 1] > wall[2, 1]
    assert np.allclose(wall[:, 0], 0.0)


def test_two_sided_wall_pushes_from_both_sides_and_past_endpoints(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    planner.set_walls([((0.0, 0.0), (3.0, 0.0))])
    pool = _pool(
        pool_empty,
        agent_factory,
        planner,
        [(1.5, 0.6, -math.pi / 2, 0.0, -0.5), (1.5, -0.6, math.pi / 2, 0.0, 0.5), (-0.6, 0.0, 0.0, 0.5, 0.0)],
        [(1.5, -10.0), (1.5, 10.0), (10.0, 0.0)],
        (np.zeros(4, dtype=np.int32), np.empty(0, dtype=np.int32)),
    )
    wall = _forces(planner, pool)[:, WALL]
    assert wall[0, 1] > 0.0
    assert wall[1, 1] < 0.0
    assert wall[0, 1] == pytest.approx(-wall[1, 1])
    assert wall[2, 0] < 0.0


def test_walls_shorter_than_guard_length_are_ignored(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    planner.set_walls([((-0.04, -0.5), (0.04, -0.5))])
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 0.0, 0.0)], [(10.0, 0.0)], (np.zeros(2, dtype=np.int32), np.empty(0, dtype=np.int32)))
    assert np.allclose(_forces(planner, pool)[0, WALL], 0.0)


def test_pooled_and_per_agent_paths_agree(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    planner.set_walls([((-5.0, -0.8), (5.0, -0.8))])
    agents = [agent_factory(agent_id=1, x=0.0, y=0.0), agent_factory(agent_id=2, x=0.9, y=0.2)]
    agents[0].state.velocity = (0.6, 0.0)
    agents[1].state.velocity = (-0.4, 0.2)
    for agent, other in zip(agents, reversed(agents), strict=True):
        agent.belief = BeliefState(observed_agents=[other.state])
    goals = {1: Pose2D(x=10.0, y=0.0), 2: Pose2D(x=-10.0, y=0.0)}

    pool = pool_empty(capacity=4)
    planner.attach(pool)
    for agent in agents:
        pool.add_agent(agent)
    pool.set_goals(goals)
    pool.set_neighbor_csr(*_PAIR)
    planner.compute_pool(pool, dt=0.05)

    per_agent = planner.compute(agents, goals, dt=0.05)
    for i, agent in enumerate(agents):
        assert np.allclose(per_agent[agent.state.agent_id], pool.vel[i])


def test_coincident_stationary_agents_on_a_wall_stay_finite(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    planner.set_walls([((-1.0, 0.0), (1.0, 0.0))])
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0)], [(5.0, 0.0), (-5.0, 0.0)], _PAIR)
    forces = _forces(planner, pool)
    assert np.all(np.isfinite(forces))
    assert np.all(np.isfinite(pool.vel[:2]))
    assert np.allclose(forces[:, SOCIAL], 0.0)
