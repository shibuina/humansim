from __future__ import annotations

from collections.abc import Callable

import attrs
import numpy as np
import pytest

from arena_humansim.core.agent_manager import arrival_latch_step
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.global_planner import GlobalPlanner
from arena_humansim.global_planner._grid import next_waypoint
from arena_humansim.local_planner.orca import (
    ORCAPlanner,
    _linear_program1,
    _linear_program2,
    _linear_program3,
)
from arena_humansim.local_planner.pedvo import PedVOPlanner
from arena_humansim.utils.benchmark import generate_maze
from arena_humansim.utils.types import Pose2D, Segments

_DT = 0.05


def _lines(*rows: tuple[float, float, float, float]) -> np.ndarray:
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def _pool(
    agent_factory: Callable[..., BaseAgent],
    specs: list[tuple[tuple[float, float], tuple[float, float] | None, tuple[float, float]]],
) -> AgentPool:
    pool = AgentPool(capacity=max(len(specs), 1))
    for i, (pos, goal, vel) in enumerate(specs):
        agent = agent_factory(agent_id=i + 1, x=pos[0], y=pos[1])
        agent.state.velocity = vel
        idx = pool.add_agent(agent)
        if goal is not None:
            pool.goal_pos[idx] = goal
            pool.has_goal[idx] = True
    return pool


def _run(planner: ORCAPlanner, pool: AgentPool, steps: int, external: dict[int, tuple[float, float]] | None = None) -> np.ndarray:
    n = pool.n
    traj = [pool.pos[:n].copy()]
    for _ in range(steps):
        planner.compute_pool(pool, dt=_DT)
        for idx, v in (external or {}).items():
            pool.vel[idx] = v
        pool.pos[:n] += pool.vel[:n] * _DT
        pool.prev_vel[:n] = pool.vel[:n]
        traj.append(pool.pos[:n].copy())
    return np.array(traj)


def _wall_clearance(points: np.ndarray, walls: Segments) -> np.ndarray:
    seg = np.asarray(walls, dtype=np.float64).reshape(-1, 4)
    a = seg[:, :2]
    ab = seg[:, 2:] - a
    p = points.reshape(-1, 1, 2)
    t = np.clip(np.sum((p - a) * ab, axis=-1) / np.sum(ab * ab, axis=-1), 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1).min(axis=1).reshape(points.shape[:-1])


def _min_pair_distance(traj: np.ndarray) -> float:
    diff = traj[:, :, None, :] - traj[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    n = traj.shape[1]
    dist[:, np.arange(n), np.arange(n)] = np.inf
    return float(dist.min())


def test_orca_no_agents_returns_empty() -> None:
    planner = ORCAPlanner()
    assert planner.compute([], {}, dt=0.1) == {}


def test_orca_missing_goal_zero_velocity(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    out = planner.compute([agent], {}, dt=0.1)
    assert out[1] == (0.0, 0.0)


def test_orca_at_goal_zero_velocity(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    agent = agent_factory(agent_id=1, x=2.0, y=2.0)
    goals = {1: Pose2D(x=2.0, y=2.0, theta=0.0)}
    out = planner.compute([agent], goals, dt=0.1)
    assert out[1] == (0.0, 0.0)


def test_orca_at_goal_with_neighbor_pressure_produces_nonzero_velocity(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner(time_horizon=5.0)
    settled = agent_factory(agent_id=1, x=0.0, y=0.0)
    pusher = agent_factory(agent_id=2, x=0.3, y=0.0)
    pusher.state.velocity = (-1.0, 0.0)
    goals = {
        1: Pose2D(x=0.0, y=0.0, theta=0.0),
        2: Pose2D(x=-10.0, y=0.0, theta=0.0),
    }
    out = planner.compute([settled, pusher], goals, dt=0.1)
    vx, vy = out[1]
    assert abs(vx) + abs(vy) > 1e-6


def test_orca_max_speed_independent_of_desired_velocity(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    agent.params = attrs.evolve(agent.params, desired_velocity=0.0, max_velocity=1.2)
    agent.state.desired_velocity = 0.0
    goals = {1: Pose2D(x=10.0, y=0.0, theta=0.0)}
    out = planner.compute([agent], goals, dt=0.1)
    vx, vy = out[1]
    assert vx == pytest.approx(0.0, abs=1e-9)
    assert vy == pytest.approx(0.0, abs=1e-9)


def test_orca_single_agent_pref_velocity_toward_goal(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    goals = {1: Pose2D(x=10.0, y=0.0, theta=0.0)}
    out = planner.compute([agent], goals, dt=0.1)
    vx, vy = out[1]
    assert vx == pytest.approx(agent.state.desired_velocity, rel=1e-6)
    assert vy == pytest.approx(0.0, abs=1e-9)


def test_orca_pref_velocity_beyond_max_speed_is_clamped(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    agent.state.desired_velocity = 3.0
    goals = {1: Pose2D(x=0.0, y=10.0, theta=0.0)}
    vx, vy = planner.compute([agent], goals, dt=0.1)[1]
    assert vx == pytest.approx(0.0, abs=1e-9)
    assert vy == pytest.approx(agent.params.max_velocity, rel=1e-9)


def test_orca_head_on_collision_cutoff_circle_branch(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner(time_horizon=5.0)
    a1 = agent_factory(agent_id=1, x=0.0, y=0.0)
    a2 = agent_factory(agent_id=2, x=2.0, y=0.0)
    a1.state.velocity = (1.0, 0.0)
    a2.state.velocity = (-1.0, 0.0)
    goals = {
        1: Pose2D(x=10.0, y=0.0, theta=0.0),
        2: Pose2D(x=-10.0, y=0.0, theta=0.0),
    }
    out = planner.compute([a1, a2], goals, dt=0.1)
    v1 = out[1]
    v2 = out[2]
    assert abs(v1[1]) + abs(v2[1]) > 1e-6 or v1[0] < a1.params.desired_velocity - 1e-6


def test_orca_leg_branch_cross_positive(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner(time_horizon=5.0)
    a1 = agent_factory(agent_id=1, x=0.0, y=0.0)
    a2 = agent_factory(agent_id=2, x=3.0, y=0.5)
    a1.state.velocity = (0.1, 0.0)
    a2.state.velocity = (0.0, 0.0)
    goals = {
        1: Pose2D(x=10.0, y=0.0, theta=0.0),
        2: Pose2D(x=10.0, y=0.5, theta=0.0),
    }
    out = planner.compute([a1, a2], goals, dt=0.1)
    assert 1 in out and 2 in out


def test_orca_leg_branch_cross_negative(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner(time_horizon=5.0)
    a1 = agent_factory(agent_id=1, x=0.0, y=0.0)
    a2 = agent_factory(agent_id=2, x=3.0, y=-0.5)
    a1.state.velocity = (0.1, 0.0)
    a2.state.velocity = (0.0, 0.0)
    goals = {
        1: Pose2D(x=10.0, y=0.0, theta=0.0),
        2: Pose2D(x=10.0, y=-0.5, theta=0.0),
    }
    out = planner.compute([a1, a2], goals, dt=0.1)
    assert 1 in out and 2 in out


def test_orca_overlapping_agents_move_apart(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner(time_horizon=5.0)
    a1 = agent_factory(agent_id=1, x=0.0, y=0.0)
    a2 = agent_factory(agent_id=2, x=0.1, y=0.0)
    a1.state.velocity = (0.5, 0.0)
    a2.state.velocity = (-0.5, 0.0)
    goals = {
        1: Pose2D(x=10.0, y=0.0, theta=0.0),
        2: Pose2D(x=-10.0, y=0.0, theta=0.0),
    }
    out = planner.compute([a1, a2], goals, dt=0.1)
    assert out[1][0] < 0.0 < out[2][0]
    for vx, vy in out.values():
        assert np.hypot(vx, vy) <= a1.params.max_velocity + 1e-9


def test_linear_program2_without_lines_clamps_pref_to_max_speed() -> None:
    fail, rx, ry = _linear_program2(_lines(), 0, 1.5, 10.0, 0.0, False)
    assert fail == 0
    assert np.hypot(rx, ry) == pytest.approx(1.5, rel=1e-9)


def test_linear_program_line_outside_speed_circle_falls_back_to_least_violation() -> None:
    lines = _lines((10.0, 0.0, 0.0, -1.0))
    fail, rx, ry = _linear_program2(lines, 1, 1.0, 2.0, 0.0, False)
    assert fail == 0
    rx, ry = _linear_program3(lines, 1, 0, fail, 1.0, rx, ry)
    assert np.hypot(rx, ry) == pytest.approx(1.0, rel=1e-9)
    assert rx == pytest.approx(1.0, rel=1e-9)


def test_linear_program_line_outside_speed_circle_with_zero_pref_still_moves_toward_it() -> None:
    lines = _lines((10.0, 0.0, 0.0, -1.0))
    fail, rx, ry = _linear_program2(lines, 1, 1.0, 0.0, 0.0, False)
    assert fail == 0
    assert (rx, ry) == (0.0, 0.0)
    rx, ry = _linear_program3(lines, 1, 0, fail, 1.0, rx, ry)
    assert rx == pytest.approx(1.0, rel=1e-9)
    assert ry == pytest.approx(0.0, abs=1e-9)


def test_linear_program1_parallel_lines_with_disjoint_sides_are_infeasible() -> None:
    lines = _lines((0.0, 0.0, 0.0, 1.0), (0.5, 0.0, 0.0, -1.0))
    ok, _, _ = _linear_program1(lines, 1, 1.0, 0.6, 0.0, False)
    assert not ok


def test_linear_program1_parallel_lines_facing_the_same_way_are_feasible() -> None:
    lines = _lines((0.0, 10.0, 0.0, -1.0), (0.5, 0.0, 0.0, -1.0))
    ok, rx, ry = _linear_program1(lines, 1, 2.0, 0.5, 0.5, False)
    assert ok
    assert rx == pytest.approx(0.5, rel=1e-9)
    assert ry == pytest.approx(0.5, rel=1e-9)


@pytest.mark.parametrize(("pref_y", "expected_y"), [(0.5, 0.5), (0.95, 0.8), (0.0, 0.2)])
def test_linear_program1_clamps_to_bounds_of_earlier_lines(pref_y: float, expected_y: float) -> None:
    lines = _lines((0.0, 0.2, 1.0, 0.0), (0.0, 0.8, -1.0, 0.0), (0.0, 0.0, 0.0, -1.0))
    ok, rx, ry = _linear_program1(lines, 2, 1.0, 0.0, pref_y, False)
    assert ok
    assert rx == pytest.approx(0.0, abs=1e-9)
    assert ry == pytest.approx(expected_y, rel=1e-9)


def test_linear_program1_single_line_projects_pref_onto_line() -> None:
    lines = _lines((0.0, 0.0, 0.0, -1.0))
    ok, rx, ry = _linear_program1(lines, 0, 2.0, 0.3, 0.7, False)
    assert ok
    assert rx == pytest.approx(0.0, abs=1e-9)
    assert ry == pytest.approx(0.7, rel=1e-9)


def test_linear_program3_keeps_obstacle_lines_hard() -> None:
    lines = _lines((0.0, 0.0, 0.0, -1.0), (-0.5, 0.0, 0.0, 1.0))
    fail, rx, ry = _linear_program2(lines, 2, 1.0, 1.0, 0.0, False)
    assert fail == 1
    rx, ry = _linear_program3(lines, 2, 1, fail, 1.0, rx, ry)
    assert rx == pytest.approx(0.0, abs=1e-9)
    assert np.hypot(rx, ry) <= 1.0 + 1e-9


def test_orca_avoids_standing_agent_without_goal(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    pool = _pool(agent_factory, [((0.0, 0.0), (6.0, 0.0), (0.0, 0.0)), ((3.0, 0.15), None, (0.0, 0.0))])
    traj = _run(planner, pool, 200)
    assert np.allclose(traj[:, 1], (3.0, 0.15))
    assert _min_pair_distance(traj) >= 0.5 - 0.05
    assert traj[-1, 0, 0] > 5.0


def test_orca_avoids_agent_of_other_planner(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    pool = _pool(agent_factory, [((0.0, 0.0), (8.0, 0.0), (0.0, 0.0)), ((8.0, 0.0), None, (-1.0, 0.0))])
    traj = _run(planner, pool, 200, external={1: (-1.0, 0.0)})
    assert _min_pair_distance(traj) >= 0.5 - 0.05
    assert traj[-1, 0, 0] > 6.0


@pytest.mark.parametrize("goal", [(0.0, 3.0), (3.0, 3.0)], ids=["perpendicular", "diagonal"])
def test_orca_agent_driven_into_wall_keeps_clearance(agent_factory: Callable[..., BaseAgent], goal: tuple[float, float]) -> None:
    walls: Segments = [((-5.0, 1.0), (5.0, 1.0))]
    planner = ORCAPlanner()
    planner.set_walls(walls)
    pool = _pool(agent_factory, [((0.0, 0.0), goal, (0.0, 0.0))])
    radius = float(pool.agent_radius[0])
    traj = _run(planner, pool, 200)
    assert _wall_clearance(traj, walls).min() >= radius - 1e-3
    assert traj[-1, 0, 1] > 0.5


def test_orca_corridor_counterflow_passes_without_penetration(agent_factory: Callable[..., BaseAgent]) -> None:
    walls: Segments = [((-10.0, -1.0), (10.0, -1.0)), ((-10.0, 1.0), (10.0, 1.0))]
    planner = ORCAPlanner()
    planner.set_walls(walls)
    pool = _pool(
        agent_factory,
        [
            ((-4.0, 0.3), (7.0, 0.3), (0.0, 0.0)),
            ((-5.0, -0.3), (6.0, -0.3), (0.0, 0.0)),
            ((4.0, 0.3), (-7.0, 0.3), (0.0, 0.0)),
            ((5.0, -0.3), (-6.0, -0.3), (0.0, 0.0)),
        ],
    )
    radius = float(pool.agent_radius[0])
    traj = _run(planner, pool, 400)
    assert _wall_clearance(traj, walls).min() >= radius - 1e-3
    assert _min_pair_distance(traj) >= 2 * radius - 0.05
    assert np.all(traj[-1, :2, 0] > 4.0)
    assert np.all(traj[-1, 2:, 0] < -4.0)


def test_orca_crowded_infeasible_case_respects_wall_and_max_speed(agent_factory: Callable[..., BaseAgent]) -> None:
    walls: Segments = [((-5.0, 0.0), (5.0, 0.0))]
    planner = ORCAPlanner(time_horizon_obst=2.0)
    planner.set_walls(walls)
    pool = _pool(
        agent_factory,
        [
            ((0.0, 0.3), (0.0, 5.0), (0.0, 0.0)),
            ((0.0, 0.65), (0.0, -5.0), (0.0, -1.0)),
            ((-0.35, 0.55), (5.0, -5.0), (0.7, -0.7)),
            ((0.35, 0.55), (-5.0, -5.0), (-0.7, -0.7)),
            ((-0.45, 0.3), (5.0, 0.3), (1.0, 0.0)),
            ((0.45, 0.3), (-5.0, 0.3), (-1.0, 0.0)),
        ],
    )
    planner.compute_pool(pool, dt=_DT)
    vx, vy = pool.vel[0]
    radius = float(pool.agent_radius[0])
    assert np.hypot(vx, vy) <= pool.max_velocity[0] + 1e-9
    assert vy >= -(0.3 - radius) / planner.time_horizon_obst - 1e-9
    for i in range(pool.n):
        assert np.hypot(*pool.vel[i]) <= pool.max_velocity[i] + 1e-9


def test_orca_compute_matches_compute_pool(agent_factory: Callable[..., BaseAgent]) -> None:
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

    planner = ORCAPlanner()
    planner.set_walls(walls)
    via_compute = planner.compute(agents, goals, dt=_DT)

    pool = AgentPool(capacity=len(agents))
    for agent in agents:
        pool.add_agent(agent)
    pool.set_goals(goals)
    planner.compute_pool(pool, dt=_DT)

    for i in range(pool.n):
        assert via_compute[int(pool.agent_ids[i])] == (float(pool.vel[i, 0]), float(pool.vel[i, 1]))


def test_orca_wall_grid_candidates_match_single_cell_full_scan(agent_factory: Callable[..., BaseAgent]) -> None:
    rng = np.random.default_rng(11)
    maze: Segments = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in generate_maze(10, seed=42).walls]
    starts = rng.uniform(-5.0, 25.0, size=(300, 2))
    angles = rng.uniform(-np.pi, np.pi, size=300)
    lengths = rng.uniform(2.0, 15.0, size=300)
    ends = starts + np.stack([np.cos(angles) * lengths, np.sin(angles) * lengths], axis=1)
    walls: Segments = maze + [((float(s[0]), float(s[1])), (float(e[0]), float(e[1]))) for s, e in zip(starts, ends, strict=True)]
    agents = []
    goals = {}
    for i in range(200):
        agent = agent_factory(agent_id=i + 1, x=float(rng.uniform(0.0, 20.0)), y=float(rng.uniform(0.0, 20.0)))
        agent.state.velocity = (float(rng.normal(0, 0.5)), float(rng.normal(0, 0.5)))
        agents.append(agent)
        goals[i + 1] = Pose2D(x=float(rng.uniform(0.0, 20.0)), y=float(rng.uniform(0.0, 20.0)), theta=0.0)

    gridded = ORCAPlanner()
    gridded.set_walls(walls)
    single_cell = ORCAPlanner(wall_grid_cell=1e9)
    single_cell.set_walls(walls)
    assert single_cell._grid.nx == single_cell._grid.ny == 1

    assert gridded.compute(agents, goals, dt=_DT) == single_cell.compute(agents, goals, dt=_DT)


@pytest.mark.parametrize("planner_cls", [ORCAPlanner, PedVOPlanner])
def test_agent_following_cornered_path_passes_every_subgoal(planner_cls: type[ORCAPlanner], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = planner_cls()
    waypoints = [Pose2D(x=0.0, y=0.0), Pose2D(x=2.0, y=0.0), Pose2D(x=2.0, y=2.0), Pose2D(x=0.0, y=2.0)]
    pool = _pool(agent_factory, [((0.0, 0.0), None, (0.0, 0.0))])
    idx = 0
    for _ in range(400):
        here = Pose2D(x=float(pool.pos[0, 0]), y=float(pool.pos[0, 1]))
        idx = GlobalPlanner.advance_along_path(here, waypoints, idx)
        sub = next_waypoint(waypoints, idx)
        pool.goal_pos[0] = (sub.x, sub.y)
        pool.has_goal[0] = True
        planner.compute_pool(pool, dt=_DT)
        pool.pos[0] += pool.vel[0] * _DT
        pool.prev_vel[0] = pool.vel[0]

    assert idx == len(waypoints) - 1
    assert np.hypot(pool.pos[0, 0] - 0.0, pool.pos[0, 1] - 2.0) < 0.15


@pytest.mark.parametrize("planner_cls", [ORCAPlanner, PedVOPlanner])
def test_latched_agent_holds_its_slot(planner_cls: type[ORCAPlanner], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = planner_cls()
    pool = _pool(agent_factory, [((0.0, 0.0), (0.1, 0.0), (0.3, 0.0)), ((3.0, 0.0), (-3.0, 0.0), (0.0, 0.0))])
    pool.set_terminals({int(pool.agent_ids[0]): Pose2D(x=0.1, y=0.0)})
    arrival_latch_step(pool, r_enter=0.15, r_exit=0.30)
    planner.compute_pool(pool, dt=_DT)

    assert bool(pool.latched[0])
    assert tuple(pool.vel[0]) == (0.0, 0.0)
