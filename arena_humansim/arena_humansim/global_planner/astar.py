from __future__ import annotations

import math
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyastar2d
from scipy.ndimage import binary_dilation

from arena_humansim.core.agents import BaseAgent
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D, Segment, Segments

from . import GlobalPlanner
from ._grid import (
    SnapMaps,
    fill_pockets,
    grid_to_world,
    nearest_free_map,
    needs_replan,
    next_waypoint,
    push_from_walls,
    simplify_with_los,
    snap_pose,
    world_to_grid,
)

_SQRT2 = math.sqrt(2)


def _nearest_free_cell(
    grid: np.ndarray,
    row: int,
    col: int,
    max_radius: int = 200,
    nearest: np.ndarray | None = None,
) -> tuple[int, int] | None:
    rows, cols = grid.shape
    if 0 <= row < rows and 0 <= col < cols and grid[row, col] == 0:
        return (row, col)
    if nearest is not None:
        r, c = min(max(row, 0), rows - 1), min(max(col, 0), cols - 1)
        return (int(nearest[0, r, c]), int(nearest[1, r, c]))
    for r in range(1, max_radius + 1):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if abs(dr) != r and abs(dc) != r:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    return (nr, nc)
    return None


def _astar_path(
    weights: np.ndarray,
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    nearest: np.ndarray | None = None,
) -> list[tuple[int, int]] | None:
    actual_start = _nearest_free_cell(grid, start[0], start[1], nearest=nearest)
    actual_goal = _nearest_free_cell(grid, goal[0], goal[1], nearest=nearest)
    if actual_start is None or actual_goal is None:
        return None

    prepend_start = actual_start != start

    if actual_start == actual_goal:
        path = [actual_start]
        if prepend_start:
            path.insert(0, start)
        return path

    result = pyastar2d.astar_path(weights, actual_start, actual_goal, allow_diagonal=True)
    if result is None:
        return None

    path = [(int(r), int(c)) for r, c in result]
    if prepend_start:
        path.insert(0, start)
    return path


class AStarPlanner(GlobalPlanner):
    #: `set_walls` takes `thin_segments` and `fallback_segments` (see `README.md`, grid parameters).
    supports_thin_segments = True

    def __init__(
        self,
        replan_distance: float = 1.0,
        inflation_radius: float = 0.38,
        resolution: float = 0.2,
        thin_inflation: float = 0.0,
    ):
        self._replan_distance = replan_distance
        self._inflation_radius = inflation_radius
        #: Metres. How far *thin* segments (small furniture) are inflated: their footprint
        #: blocks the grid so paths go round them, but they do not seal the doorway they
        #: stand behind the way a full agent-radius inflation does.
        self._thin_inflation = thin_inflation

        self._occupancy_grid: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._nearest: np.ndarray | None = None
        self._snap = SnapMaps()
        #: The walls alone, uninflated: what a snap may not walk through.
        self._walls_grid: np.ndarray | None = None
        #: Walls-only grid: where furniture seals a room off in the main grid, the route is
        #: planned here and the local planner squeezes past the furniture.
        self._fallback_grid: np.ndarray | None = None
        self._fallback_weights: np.ndarray | None = None
        self._fallback_nearest: np.ndarray | None = None
        self._resolution: float = resolution
        self._origin: Pose2D = Pose2D()
        self._wall_segments: list[Segment] = []

        self._path_cache: dict[int, tuple[tuple[float, float], list[Pose2D], int]] = {}
        self._cached_results: dict[int, Pose2D] = {}
        self._pool = ThreadPoolExecutor(max_workers=max((os.cpu_count() or 2) - 1, 1))

    def set_walls(self, segments: Segments, thin_segments: Segments = (), fallback_segments: Segments | None = None) -> None:
        """Rasterise `segments` inflated by `inflation_radius` and `thin_segments` by
        `thin_inflation` into one grid. Only `segments` push waypoints back. With
        `fallback_segments` (the walls alone) a second grid is kept for routes the furnished
        grid cannot provide."""
        self._path_cache.clear()
        self._weights = None
        self._nearest = None
        self._snap = SnapMaps()
        self._walls_grid = None
        self._fallback_grid = self._fallback_weights = self._fallback_nearest = None
        self._wall_segments = list(segments)
        thin_segments = list(thin_segments)
        if not segments and not thin_segments:
            self._occupancy_grid = None
            return

        arr = np.array([*segments, *thin_segments], dtype=np.float64).reshape(-1, 2, 2)
        all_points = arr.reshape(-1, 2)
        margin = self._inflation_radius + self._resolution * 2
        x_min = float(all_points[:, 0].min()) - margin
        y_min = float(all_points[:, 1].min()) - margin
        x_max = float(all_points[:, 0].max()) + margin
        y_max = float(all_points[:, 1].max()) + margin

        self._origin = Pose2D(x=x_min, y=y_min)
        res = self._resolution
        cols = int(math.ceil((x_max - x_min) / res)) + 1
        rows = int(math.ceil((y_max - y_min) / res)) + 1

        def rasterize(segs: Segments, inflation: float) -> np.ndarray:
            g = np.zeros((rows, cols), dtype=np.uint8)
            for (x1, y1), (x2, y2) in segs:
                c1, r1 = (x1 - x_min) / res, (y1 - y_min) / res
                c2, r2 = (x2 - x_min) / res, (y2 - y_min) / res
                # Two samples per cell along the longer axis, so a segment leaves no one-cell gaps.
                n = int(math.ceil(2.0 * max(abs(c2 - c1), abs(r2 - r1)))) + 1
                for t in np.linspace(0.0, 1.0, n):
                    c = int(round(c1 + t * (c2 - c1)))
                    r = int(round(r1 + t * (r2 - r1)))
                    if 0 <= r < rows and 0 <= c < cols:
                        g[r, c] = 1
            k = int(math.ceil(inflation / res))
            if k > 0:
                y, x = np.ogrid[-k : k + 1, -k : k + 1]
                g = binary_dilation(g, structure=(x * x + y * y) <= k * k).astype(np.uint8)
            return g

        radius_cells = int(math.ceil(self._inflation_radius / res))
        grid = rasterize(segments, self._inflation_radius)
        if thin_segments:
            grid = (grid | rasterize(thin_segments, self._thin_inflation)).astype(np.uint8)
        grid, pockets = fill_pockets(grid)

        self._occupancy_grid = grid
        self._weights = np.where(grid == 0, 1.0, np.inf).astype(np.float32)
        self._nearest = nearest_free_map(grid)
        if fallback_segments is not None:
            fb = rasterize(list(fallback_segments), self._inflation_radius)
            self._fallback_grid = fb
            self._walls_grid = rasterize(list(fallback_segments), 0.0)
            self._fallback_weights = np.where(fb == 0, 1.0, np.inf).astype(np.float32)
            self._fallback_nearest = nearest_free_map(fb)
        thin_note = f" + {len(thin_segments)} thin at {self._thin_inflation}m" if thin_segments else ""
        thin_note += f", {pockets} sealed pocket(s) filled" if pockets else ""
        self._logger.info(f"Walls rasterized: {cols}x{rows} ({cols * rows} cells), res={res}m, {len(segments)} segment(s), inflation={self._inflation_radius}m ({radius_cells} cells){thin_note}")

    def get_cached_goals(self) -> dict[int, Pose2D]:
        return dict(self._cached_results)

    def get_cached_paths(self) -> dict[int, list[Pose2D]]:
        return {aid: wps for aid, (_, wps, _) in self._path_cache.items()}

    def invalidate_paths(self, agent_ids: Iterable[int]) -> None:
        for aid in agent_ids:
            self._path_cache.pop(aid, None)

    def snap_terminal(self, pose: Pose2D) -> Pose2D:
        return snap_pose(self._occupancy_grid, self._origin, self._resolution, self._inflation_radius, pose, 0.0, self._snap, self._walls_grid)

    def snap_spawn(self, pose: Pose2D, radius: float) -> Pose2D:
        return snap_pose(self._occupancy_grid, self._origin, self._resolution, self._inflation_radius, pose, radius, self._snap, self._walls_grid)

    def compute(
        self,
        agents: Iterable[BaseAgent],
        high_level_commands: dict[int, HighLevelCommand],
    ) -> dict[int, Pose2D]:
        agent_positions: dict[int, Pose2D] = {agent.state.agent_id: agent.state.pose for agent in agents}
        goals: dict[int, Pose2D] = {}
        has_grid = self._occupancy_grid is not None and self._weights is not None

        replan_requests: list[tuple[int, Pose2D, Pose2D, tuple[int, int], tuple[int, int]]] = []

        for agent_id, cmd in high_level_commands.items():
            if not isinstance(cmd, HighLevelCommand):
                continue
            if cmd.type != CommandType.NAVIGATE:
                continue

            target = cmd.target_pose
            agent_pos = agent_positions.get(agent_id)

            if agent_pos is None or not has_grid:
                goals[agent_id] = target
                continue

            if not needs_replan(self._path_cache, agent_id, target, agent_pos, self._replan_distance):
                cached_goal, waypoints, idx = self._path_cache[agent_id]
                idx = self.advance_along_path(agent_pos, waypoints, idx)
                self._path_cache[agent_id] = (cached_goal, waypoints, idx)
                goals[agent_id] = next_waypoint(waypoints, idx)
                continue

            start_rc = world_to_grid(self._origin, self._resolution, agent_pos.x, agent_pos.y)
            # the planner's own goal snap is by straight line; hand it the walked one
            snapped_target = self.snap_terminal(target)
            goal_rc = world_to_grid(self._origin, self._resolution, snapped_target.x, snapped_target.y)
            replan_requests.append((agent_id, agent_pos, target, start_rc, goal_rc))

        if replan_requests:
            weights = self._weights
            grid = self._occupancy_grid
            futures = {agent_id: self._pool.submit(_astar_path, weights, grid, start_rc, goal_rc, self._nearest) for agent_id, _, _, start_rc, goal_rc in replan_requests}

            for agent_id, agent_pos, target, start_rc, goal_rc in replan_requests:
                raw_path = futures[agent_id].result()

                if raw_path is None and self._fallback_grid is not None:
                    raw_path = _astar_path(self._fallback_weights, self._fallback_grid, start_rc, goal_rc, self._fallback_nearest)
                    if raw_path is not None:
                        self._logger.debug(f"Furniture seals agent {agent_id}'s route ({start_rc} -> {goal_rc}); walls-only route")

                if raw_path is None:
                    self._logger.debug(f"No path for agent {agent_id} ({start_rc} -> {goal_rc}), using direct goal")
                    goals[agent_id] = self.snap_terminal(target)
                    self._path_cache.pop(agent_id, None)
                    continue

                waypoints = [grid_to_world(self._origin, self._resolution, r, c) for r, c in raw_path]
                waypoints[0] = Pose2D(x=agent_pos.x, y=agent_pos.y)
                waypoints[-1] = self.snap_terminal(target)
                assert self._occupancy_grid is not None
                waypoints = simplify_with_los(self._occupancy_grid, self._origin, self._resolution, waypoints)
                if self._wall_segments:
                    waypoints = push_from_walls(self._wall_segments, self._inflation_radius, waypoints)

                goal_key = (round(target.x, 3), round(target.y, 3))
                idx = self.advance_along_path(agent_pos, waypoints, 0)
                self._path_cache[agent_id] = (goal_key, waypoints, idx)
                goals[agent_id] = next_waypoint(waypoints, idx)

        self._cached_results = goals
        return goals
