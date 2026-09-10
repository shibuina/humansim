from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.pool import PoolAware
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import HighLevelCommand, Pose2D, WallAware

if TYPE_CHECKING:
    from arena_humansim.core.viz import MarkerPublisher

_registry: ModuleRegistry[GlobalPlanner] = ModuleRegistry()


def simplify_path(
    waypoints: Sequence[Pose2D],
    min_area: float = 0.01,
) -> list[Pose2D]:
    n = len(waypoints)
    if n <= 2:
        return list(waypoints)

    xs = [w.x for w in waypoints]
    ys = [w.y for w in waypoints]
    prev_idx = list(range(-1, n - 1))
    next_idx = list(range(1, n + 1))

    _inf = math.inf
    _heappush = heapq.heappush
    _heappop = heapq.heappop

    def _area(i: int) -> float:
        p, nx = prev_idx[i], next_idx[i]
        return 0.5 * abs((xs[i] - xs[p]) * (ys[nx] - ys[p]) - (xs[nx] - xs[p]) * (ys[i] - ys[p]))

    areas = [_inf] + [_area(i) for i in range(1, n - 1)] + [_inf]
    heap = [(areas[i], i) for i in range(1, n - 1)]
    heapq.heapify(heap)
    removed = bytearray(n)

    while heap:
        area, i = _heappop(heap)
        if removed[i] or areas[i] != area:
            continue
        if area >= min_area:
            break

        removed[i] = 1
        p, nx = prev_idx[i], next_idx[i]
        next_idx[p] = nx
        prev_idx[nx] = p

        if p > 0:
            a = max(_area(p), area)
            areas[p] = a
            _heappush(heap, (a, p))
        if nx < n - 1:
            a = max(_area(nx), area)
            areas[nx] = a
            _heappush(heap, (a, nx))

    return [waypoints[i] for i in range(n) if not removed[i]]


#: Metres. An intermediate path waypoint counts as reached inside this radius (below the
#: 0.38 m wall inflation, so cutting the corner never crosses a wall).
WAYPOINT_REACH_M = 0.3


class GlobalPlanner(PoolAware, WallAware, Loggable, ABC):
    @abstractmethod
    def compute(
        self,
        agents: Iterable[BaseAgent],
        high_level_commands: dict[int, HighLevelCommand],
    ) -> dict[int, Pose2D]: ...

    @abstractmethod
    def get_cached_goals(self) -> dict[int, Pose2D]: ...

    def get_cached_paths(self) -> dict[int, list[Pose2D]]:
        return {}

    def invalidate_paths(self, agent_ids: Iterable[int]) -> None:
        pass

    def snap_terminal(self, pose: Pose2D) -> Pose2D:
        return pose

    def snap_spawn(self, pose: Pose2D, radius: float) -> Pose2D:
        """Where an agent of `radius` spawning at `pose` should stand: `pose` itself when it
        is clear of the furnished grid, otherwise the nearest cell with room for its body."""
        return pose

    def publish_markers(self, pub: MarkerPublisher) -> None:
        pass

    @staticmethod
    def advance_along_path(
        agent_pos: Pose2D,
        waypoints: Sequence[Pose2D],
        current_idx: int,
        reach: float = WAYPOINT_REACH_M,
    ) -> int:
        """Index of the path segment the agent is on.

        A segment is left when the agent has passed its end point along the segment's
        direction, *or* stands within `reach` of that point: the local planner never lands
        exactly on an intermediate waypoint, and one pushed out to the wall-inflation margin
        sits where wall repulsion holds the agent a few centimetres short. Without the reach
        the attraction force collapses on that point and the agent stands there for good.
        """
        idx = current_idx
        reach_sq = reach * reach
        while idx < len(waypoints) - 1:
            wp = waypoints[idx]
            nxt = waypoints[idx + 1]
            dx, dy = nxt.x - wp.x, nxt.y - wp.y
            tx, ty = agent_pos.x - wp.x, agent_pos.y - wp.y
            nx, ny = agent_pos.x - nxt.x, agent_pos.y - nxt.y
            if dx * tx + dy * ty >= dx * dx + dy * dy or nx * nx + ny * ny <= reach_sq:
                idx += 1
            else:
                break
        return idx

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[], type[GlobalPlanner]]], Callable[[], type[GlobalPlanner]]]:
        return _registry.register(name)

    @classmethod
    def get_class(cls, name: str) -> type[GlobalPlanner]:
        return _registry.get(name)

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> GlobalPlanner:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()


def _load_dijkstra() -> type[GlobalPlanner]:
    from .dijkstra import DijkstraPlanner

    return DijkstraPlanner


def _load_astar() -> type[GlobalPlanner]:
    from .astar import AStarPlanner

    return AStarPlanner


_registry.register("dijkstra")(_load_dijkstra)
_registry.register("astar")(_load_astar)
