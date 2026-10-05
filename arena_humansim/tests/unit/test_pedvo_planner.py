from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.orca import ORCAPlanner, _linear_program1, _linear_program2, _linear_program3
from arena_humansim.local_planner.pedvo import (
    PedVOPlanner,
    _linear_program1_turning,
    _linear_program2_turning,
    _linear_program3_turning,
)
from arena_humansim.utils.types import Pose2D, Segments

_DT = 0.05


def _pool(
    agent_factory: Callable[..., BaseAgent],
    specs: list[tuple[tuple[float, float], tuple[float, float] | None, tuple[float, float]]],
) -> AgentPool:
    pool = AgentPool(capacity=max(len(specs), 1))
    for i, (pos, goal, vel) in enumerate(specs):
        agent = agent_factory(agent_id=i + 1, x=pos[0], y=pos[1])
        agent.state.velocity = vel
        idx = pool.add_agent(agent)
        pool.prev_vel[idx] = vel
        if goal is not None:
            pool.goal_pos[idx] = goal
            pool.has_goal[idx] = True
    return pool


def _run(planner: ORCAPlanner, pool: AgentPool, steps: int) -> tuple[np.ndarray, np.ndarray]:
    n = pool.n
    traj = [pool.pos[:n].copy()]
    vels = []
    for _ in range(steps):
        planner.compute_pool(pool, dt=_DT)
        vels.append(pool.vel[:n].copy())
        pool.pos[:n] += pool.vel[:n] * _DT
        pool.prev_vel[:n] = pool.vel[:n]
        traj.append(pool.pos[:n].copy())
    return np.array(traj), np.array(vels)


def _wall_clearance(points: np.ndarray, walls: Segments) -> np.ndarray:
    seg = np.asarray(walls, dtype=np.float64).reshape(-1, 4)
    a = seg[:, :2]
    ab = seg[:, 2:] - a
    p = points.reshape(-1, 1, 2)
    t = np.clip(np.sum((p - a) * ab, axis=-1) / np.sum(ab * ab, axis=-1), 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1).min(axis=1).reshape(points.shape[:-1])


def _random_lines(rng: np.random.Generator, n: int) -> np.ndarray:
    lines = np.empty((n, 4), dtype=np.float64)
    lines[:, :2] = rng.uniform(-2.0, 2.0, size=(n, 2))
    ang = rng.uniform(-np.pi, np.pi, size=n)
    lines[:, 2] = np.cos(ang)
    lines[:, 3] = np.sin(ang)
    return lines


def test_turning_linear_programs_with_unit_bias_match_orca() -> None:
    rng = np.random.default_rng(3)
    for _ in range(500):
        n = int(rng.integers(1, 12))
        lines = _random_lines(rng, n)
        radius = float(rng.uniform(0.5, 2.0))
        opt_x, opt_y = (float(v) for v in rng.uniform(-2.5, 2.5, size=2))
        for line_no in range(n):
            for direction_opt in (False, True):
                assert _linear_program1_turning(lines, line_no, radius, opt_x, opt_y, direction_opt, 1.0) == _linear_program1(lines, line_no, radius, opt_x, opt_y, direction_opt)
        fail, rx, ry = _linear_program2_turning(lines, n, radius, opt_x, opt_y, False, 1.0)
        assert (fail, rx, ry) == _linear_program2(lines, n, radius, opt_x, opt_y, False)
        if fail < n:
            n_obst = int(rng.integers(0, fail + 1))
            assert _linear_program3_turning(lines, n, n_obst, fail, radius, rx, ry, 1.0) == _linear_program3(lines, n, n_obst, fail, radius, rx, ry)


@pytest.mark.parametrize("turn_bias", [0.3, 0.7, 2.0, 5.0])
def test_turning_linear_programs_stay_finite_for_any_bias(turn_bias: float) -> None:
    rng = np.random.default_rng(5)
    for _ in range(300):
        n = int(rng.integers(1, 12))
        lines = _random_lines(rng, n)
        fail, rx, ry = _linear_program2_turning(lines, n, 1.5, 1.2, 0.0, False, turn_bias)
        if fail < n:
            rx, ry = _linear_program3_turning(lines, n, 0, fail, 1.5, rx, ry, turn_bias)
        assert math.isfinite(rx)
        assert math.isfinite(ry)


def test_pedvo_head_on_matches_orca_with_same_horizons(agent_factory: Callable[..., BaseAgent]) -> None:
    specs = [((0.0, 0.0), (10.0, 0.05), (0.0, 0.0)), ((10.0, 0.0), (0.0, -0.05), (0.0, 0.0))]
    pedvo_traj, _ = _run(PedVOPlanner(), _pool(agent_factory, specs), 200)
    orca_traj, _ = _run(ORCAPlanner(time_horizon=2.5, time_horizon_obst=0.15), _pool(agent_factory, specs), 200)
    np.testing.assert_allclose(pedvo_traj, orca_traj, rtol=0.0, atol=1e-12)
    gaps = np.linalg.norm(pedvo_traj[:, 0] - pedvo_traj[:, 1], axis=-1)
    assert gaps.min() >= 0.5 - 0.05


def _density_cap(planner: PedVOPlanner, pos: np.ndarray, others: np.ndarray, pref_dir: np.ndarray) -> float:
    crit = pos + pref_dir
    disp = others - crit
    along = disp @ pref_dir
    stretched = (disp - along[:, None] * pref_dir) * 2.5 + along[:, None] * pref_dir
    norm = 1.0 / (1.5 * math.sqrt(2.0 * math.pi))
    density = float(np.sum(norm * np.exp(-np.sum(stretched**2, axis=1) / (2.0 * 1.5**2))))
    stride_const = 0.5 * (1.0 + planner.stride_buffer) / planner.stride_factor
    return (0.48 / density) ** 2 / stride_const**2


def test_crowd_in_front_reduces_preferred_speed(agent_factory: Callable[..., BaseAgent]) -> None:
    crowd = [(0.8, 0.0), (1.3, 0.0), (1.05, 0.5), (1.05, -0.5), (1.8, 0.0)]
    specs = [((0.0, 0.0), (20.0, 0.0), (1.3, 0.0))] + [(c, None, (1.3, 0.0)) for c in crowd]

    pool = _pool(agent_factory, specs)
    planner = PedVOPlanner()
    planner.compute_pool(pool, dt=_DT)
    expected = _density_cap(planner, np.zeros(2), np.array(crowd), np.array([1.0, 0.0]))
    assert expected < 0.6
    assert pool.vel[0, 0] == pytest.approx(expected, rel=1e-9)
    assert pool.vel[0, 1] == pytest.approx(0.0, abs=1e-9)

    unaware = _pool(agent_factory, specs)
    PedVOPlanner(dense_aware=False).compute_pool(unaware, dt=_DT)
    assert unaware.vel[0, 0] == pytest.approx(float(pool.desired_vel[0]), rel=1e-9)


def test_empty_scene_keeps_preferred_speed(agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _pool(agent_factory, [((0.0, 0.0), (0.0, 20.0), (0.0, 0.0))])
    PedVOPlanner().compute_pool(pool, dt=_DT)
    assert pool.vel[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert pool.vel[0, 1] == pytest.approx(float(pool.desired_vel[0]), rel=1e-12)


def test_walls_in_front_reduce_preferred_speed_and_walls_behind_do_not(agent_factory: Callable[..., BaseAgent]) -> None:
    xs = (0.8, 1.0, 1.2, 1.4, 1.6)
    walls: Segments = [((x, -1.0), (x, 1.0)) for x in xs]
    planner = PedVOPlanner()
    planner.set_walls(walls)

    away = _pool(agent_factory, [((0.0, 0.0), (-20.0, 0.0), (0.0, 0.0))])
    planner.compute_pool(away, dt=_DT)
    assert away.vel[0, 0] == pytest.approx(-float(away.desired_vel[0]), rel=1e-12)

    facing = _pool(agent_factory, [((0.0, 0.0), (20.0, 0.0), (0.0, 0.0))])
    planner.compute_pool(facing, dt=_DT)
    norm = 1.0 / (1.5 * math.sqrt(2.0 * math.pi))
    density = sum(norm * math.exp(-((x - 1.0) ** 2) / (2.0 * 0.75**2)) for x in xs)
    stride_const = 0.5 * (1.0 + planner.stride_buffer) / planner.stride_factor
    expected = (0.48 / density) ** 2 / stride_const**2
    assert expected < 0.9 * float(facing.desired_vel[0])
    assert facing.vel[0, 0] == pytest.approx(expected, rel=1e-9)
    assert facing.vel[0, 1] == pytest.approx(0.0, abs=1e-12)


def _max_heading_deviation(vels: np.ndarray, agent: int, axis: np.ndarray) -> float:
    v = vels[:, agent]
    speed = np.linalg.norm(v, axis=1)
    moving = speed > 1e-6
    cos = np.clip((v[moving] @ axis) / speed[moving], -1.0, 1.0)
    return float(np.arccos(cos).max())


def test_turning_bias_orders_heading_change_against_speed_change(agent_factory: Callable[..., BaseAgent]) -> None:
    specs = [((-3.0, 0.0), (8.0, 0.0), (1.3, 0.0)), ((0.0, -3.0), (0.0, 8.0), (0.0, 1.3))]
    axes = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    deviations = []
    min_speeds = []
    for turning_bias in (0.5, 1.0, 3.0):
        traj, vels = _run(PedVOPlanner(turning_bias=turning_bias), _pool(agent_factory, specs), 160)
        deviations.append([_max_heading_deviation(vels, agent, axes[agent]) for agent in range(2)])
        min_speeds.append(float(np.linalg.norm(vels[:, 0], axis=-1).min()))
        assert np.linalg.norm(traj[:, 0] - traj[:, 1], axis=-1).min() >= 0.5 - 0.05
        assert np.all(np.linalg.norm(vels, axis=-1) <= 1.5 + 1e-9)
        assert traj[-1, 0, 0] > 4.0
        assert traj[-1, 1, 1] > 4.0
    for agent in range(2):
        assert deviations[0][agent] < deviations[1][agent] < deviations[2][agent]
    assert min_speeds[0] < min_speeds[1] < min_speeds[2]


@pytest.mark.parametrize("turning_bias", [1.0, 2.5])
def test_pedvo_compute_matches_compute_pool(agent_factory: Callable[..., BaseAgent], turning_bias: float) -> None:
    walls: Segments = [((-3.0, -3.0), (3.0, -3.0)), ((3.0, -3.0), (3.0, 3.0)), ((-1.0, 0.0), (1.0, 0.5))]
    rng = np.random.default_rng(7)
    agents = []
    goals = {}
    for i in range(40):
        agent = agent_factory(agent_id=i + 1, x=float(rng.uniform(-4, 4)), y=float(rng.uniform(-4, 4)))
        agent.state.velocity = (float(rng.normal(0, 0.5)), float(rng.normal(0, 0.5)))
        agents.append(agent)
        if i % 5:
            goals[i + 1] = Pose2D(x=float(rng.uniform(-4, 4)), y=float(rng.uniform(-4, 4)), theta=0.0)

    planner = PedVOPlanner(turning_bias=turning_bias)
    planner.set_walls(walls)
    via_compute = planner.compute(agents, goals, dt=_DT)

    pool = AgentPool(capacity=len(agents))
    for agent in agents:
        pool.add_agent(agent)
    pool.set_goals(goals)
    planner.compute_pool(pool, dt=_DT)

    for i in range(pool.n):
        assert via_compute[int(pool.agent_ids[i])] == (float(pool.vel[i, 0]), float(pool.vel[i, 1]))


@pytest.mark.parametrize("turning_bias", [1.0, 3.0])
@pytest.mark.parametrize("goal", [(0.0, 3.0), (3.0, 3.0)], ids=["perpendicular", "diagonal"])
def test_pedvo_agent_driven_into_wall_does_not_cross(agent_factory: Callable[..., BaseAgent], goal: tuple[float, float], turning_bias: float) -> None:
    walls: Segments = [((-5.0, 1.0), (5.0, 1.0))]
    planner = PedVOPlanner(turning_bias=turning_bias)
    planner.set_walls(walls)
    pool = _pool(agent_factory, [((0.0, 0.0), goal, (0.0, 0.0))])
    radius = float(pool.agent_radius[0])
    traj, _ = _run(planner, pool, 300)
    assert np.all(traj[:, 0, 1] < 1.0)
    assert _wall_clearance(traj, walls).min() >= radius - 1e-3
    assert traj[-1, 0, 1] > 0.5


@pytest.mark.parametrize("turning_bias", [0.5, 1.0, 3.0])
def test_coincident_agents_produce_finite_bounded_velocities(agent_factory: Callable[..., BaseAgent], turning_bias: float) -> None:
    specs = [
        ((1.0, 1.0), (5.0, 1.0), (0.0, 0.0)),
        ((1.0, 1.0), (-5.0, 1.0), (0.0, 0.0)),
        ((1.0, 1.0), (1.0, 5.0), (0.3, 0.0)),
        ((1.0, 1.0), None, (0.0, 0.0)),
    ]
    planner = PedVOPlanner(turning_bias=turning_bias)
    planner.set_walls([((1.0, 0.0), (1.0, 2.0))])
    pool = _pool(agent_factory, specs)
    _, vels = _run(planner, pool, 20)
    assert np.all(np.isfinite(vels))
    assert np.all(np.linalg.norm(vels, axis=-1) <= pool.max_velocity[: pool.n] + 1e-9)
