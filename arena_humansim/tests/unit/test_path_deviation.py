from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner._grid import min_distances_to_paths
from arena_humansim.global_planner.navmesh import NavMeshPlanner
from arena_humansim.utils.types import Segments

_INFLATION = 0.38
_REPLAN = 1.0


def _scalar_min_distance(pos: tuple[float, float], waypoints: np.ndarray) -> float:
    best = math.inf
    prev: tuple[float, float] | None = None
    for wx, wy in waypoints.tolist():
        ax, ay = (wx, wy) if prev is None else prev
        dx, dy = wx - ax, wy - ay
        span = dx * dx + dy * dy
        t = 0.0 if span == 0.0 else max(0.0, min(1.0, ((pos[0] - ax) * dx + (pos[1] - ay) * dy) / span))
        d = math.hypot(pos[0] - ax - t * dx, pos[1] - ay - t * dy)
        if d < best:
            best = d
        prev = (wx, wy)
    return best


def _random_path(rng: np.random.Generator, length: int) -> np.ndarray:
    path = rng.uniform(-10.0, 10.0, size=(length, 2))
    if length > 2:
        repeat = rng.integers(1, length, size=max(1, length // 5))
        path[repeat] = path[repeat - 1]
    return path


def test_batched_distances_match_scalar_reference() -> None:
    rng = np.random.default_rng(1234)
    paths = [_random_path(rng, length) for length in rng.choice([1, 2, 50], size=300)]
    positions = rng.uniform(-12.0, 12.0, size=(len(paths), 2))
    on_waypoint = rng.choice(len(paths), size=60, replace=False)
    for k in on_waypoint:
        positions[k] = paths[k][rng.integers(len(paths[k]))]

    batched = min_distances_to_paths(positions, paths)

    expected = np.array([_scalar_min_distance((float(p[0]), float(p[1])), path) for p, path in zip(positions, paths, strict=True)])
    assert batched.shape == (len(paths),)
    np.testing.assert_allclose(batched, expected, rtol=0.0, atol=1e-12)
    assert np.all(batched[on_waypoint] == 0.0)


def test_batched_distance_counts_first_waypoint_as_point() -> None:
    paths = [np.array([[0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]]), np.array([[1.0, 1.0], [3.0, 1.0]])]
    positions = np.array([[3.0, 4.0], [2.0, -1.5], [0.0, 1.0]])
    np.testing.assert_allclose(min_distances_to_paths(positions, paths), [5.0, 1.5, 1.0], rtol=0.0, atol=1e-12)


def _open_room() -> Segments:
    return [
        ((0.0, 0.0), (20.0, 0.0)),
        ((20.0, 0.0), (20.0, 10.0)),
        ((20.0, 10.0), (0.0, 10.0)),
        ((0.0, 10.0), (0.0, 0.0)),
    ]


@pytest.fixture
def planned(
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> NavMeshPlanner:
    planner = NavMeshPlanner(replan_distance=_REPLAN, inflation_radius=_INFLATION)
    planner.set_walls(_open_room())
    agents = [agent_factory(agent_id=i, x=2.0, y=2.0 + 2.0 * i) for i in range(3)]
    planner.compute(agents, commands_factory(agent_ids=[0, 1, 2], target=(18.0, 5.0)))
    assert set(planner.get_cached_paths()) == {0, 1, 2}
    return planner


def test_agent_near_cached_path_keeps_path_and_far_agent_replans(
    planned: NavMeshPlanner,
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    before = planned.get_cached_paths()
    agents = [
        agent_factory(agent_id=0, x=2.0, y=2.0 + 0.5 * _REPLAN),
        agent_factory(agent_id=1, x=2.0, y=4.0),
        agent_factory(agent_id=2, x=2.0, y=6.0 + 2.0 * _REPLAN),
    ]
    planned.compute(agents, commands_factory(agent_ids=[0, 1, 2], target=(18.0, 5.0)))
    after = planned.get_cached_paths()
    assert after[0] is before[0]
    assert after[1] is before[1]
    assert after[2] is not before[2]
    assert math.hypot(after[2][0].x - 2.0, after[2][0].y - (6.0 + 2.0 * _REPLAN)) < 1e-6


def test_changed_goal_replans_agent_on_cached_path(
    planned: NavMeshPlanner,
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    before = planned.get_cached_paths()
    agents = [agent_factory(agent_id=i, x=2.0, y=2.0 + 2.0 * i) for i in range(3)]
    cmds = commands_factory(agent_ids=[0, 1], target=(18.0, 5.0)) | commands_factory(agent_ids=[2], target=(18.0, 8.0))
    planned.compute(agents, cmds)
    after = planned.get_cached_paths()
    assert after[0] is before[0]
    assert after[1] is before[1]
    assert after[2] is not before[2]
    assert math.hypot(after[2][-1].x - 18.0, after[2][-1].y - 8.0) < 1e-6
