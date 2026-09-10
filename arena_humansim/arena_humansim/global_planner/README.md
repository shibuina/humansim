# Global planners

Turn a `HighLevelCommand` (usually `NAVIGATE target_pose`) into the next global subgoal for each agent. Runs before the local planner each tick. Responsible for pathfinding, LOS simplification, and wall push-back.

## Available

| Name | Class | Notes |
|---|---|---|
| `dijkstra` | `DijkstraPlanner` | All-pairs shortest-path on an inflated occupancy grid, cached until walls change. Default. |
| `astar` | `AStarPlanner` | Per-agent A* on the same grid. Cheaper for small crowds, scales worse than cached Dijkstra as N grows. |

## Contract

```python
class GlobalPlanner(WallAware, Loggable, ABC):
    @abstractmethod
    def compute(self, agents, high_level_commands) -> dict[int, Pose2D]: ...

    @abstractmethod
    def get_cached_goals(self) -> dict[int, Pose2D]: ...

    def get_cached_paths(self) -> dict[int, list[Pose2D]]: ...
```

- `compute` returns the *next immediate subgoal* per agent - not the final target. Advancement along the cached path is handled by `advance_along_path` (shared base-class helper) using dot-product projection.
- `get_cached_goals` is how the rest of the pipeline (local planner, markers) reads the planner state without re-triggering work.
- `get_cached_paths` is optional; only used for visualization.
- `WallAware.set_walls(segments)` invalidates any grid / graph caches.

## Grid parameters (`astar`)

Node parameters, defaults unchanged:

| Parameter | Default | Meaning |
|---|---|---|
| `global_planner_inflation` | 0.38 | metres that walls and full obstacles are inflated by |
| `global_planner_resolution` | 0.2 | grid cell size in metres; the inflation is quantised to whole cells |
| `global_planner_min_obstacle_extent` | 0.0 | obstacles whose longer side is below this are *thin*: rasterised at their footprint inflated by `global_planner_thin_inflation` only, so they block the grid without sealing the doorway they stand behind. Also enables the walls-only fallback route. `0` inflates every obstacle fully. |
| `global_planner_thin_inflation` | 0.0 | metres that thin obstacles are inflated by |

At the default 0.38 m / 0.2 m grid a furnished room seals easily: a chair behind a doorway plus the inflation closes it, and every agent routed through that door ends up standing against a wall. Arena's launch file sets 0.25 m / 0.1 m, treats furniture under 0.7 m as thin at 0.15 m, and enables the fallback.

How the grid planners handle furniture:

- Thin obstacles are rasterised at their footprint only. They are also soft for the local planner and the collision resolver, which see walls and full obstacles only: the global plan routes round a chair, and a chair in a doorway is walked through rather than held against.
- When furniture still seals a route, the planner falls back to a walls-only grid and the local planner squeezes past.
- Free regions smaller than `POCKET_FRACTION` (25 %) of the largest one are pockets the inflation has cut off. They count as occupied, so a spawn or waypoint in one snaps to the nearest reachable cell instead of taking a fallback route through the furniture that seals it. Regions of comparable size (two halves of a world without a door) are left alone.
- Spawns inside furniture are moved. `AgentManager` passes every pedestrian spawn through `snap_spawn(pose, radius)`: a pose whose cell is occupied, or closer than `radius + SPAWN_MARGIN_M` (0.1 m) to the occupied region, becomes the nearest cell with that room. The move is logged with its distance.
- The snap walks. `snap_pose` takes the first fitting cell a 4-connected walk over the uninflated walls grid reaches (`nearest_by_walk`, capped at `WALK_LIMIT_CELLS`), so a point in a sealed room leaves through its door rather than through a wall. The straight-line nearest cell is the fallback when no walls grid is set (`dijkstra`) or the walk finds nothing. `compute` hands A* the walked goal cell. Walk results are cached per cell for the grid's lifetime.
- `advance_along_path` counts a waypoint as reached within `WAYPOINT_REACH_M` (0.3 m); the local planner never lands exactly on an intermediate waypoint.
- The nearest-free-cell lookup is a distance transform computed once per grid.

## Shared helpers

- `simplify_path(waypoints, min_area=0.01)` - Visvalingam-Whyatt decimation, in the base module. Use this before returning paths so LOS-connected runs collapse to endpoints.
- `GlobalPlanner.advance_along_path(agent_pos, waypoints, current_idx)` - dot-product projection; returns the new cursor index.
- `_grid.fill_pockets(grid, fraction)` - the pocket fill above, applied by both grid planners after rasterising.
- `_grid.snap_pose(grid, origin, resolution, inflation, pose, radius, cache, passable=None)`, `nearest_by_walk`, `nearest_free_map(grid)` - the snap behind `snap_spawn` (radius = the body) and `snap_terminal` (the goal).

## Adding a planner

1. Subclass `GlobalPlanner` in a new file under `global_planner/`.
2. Implement `compute` + `get_cached_goals`. Use `simplify_path` for LOS cleanup.
3. Register in `global_planner/__init__.py` via a `_load_<name>` lazy loader.
4. Add contract coverage in `tests/contracts/test_global_planner_contract.py` and efficacy coverage in `tests/efficacy/test_global_planner_efficacy.py`.

## Benchmarks

See [config/benchmark/astar_vs_dijkstra.yaml](../../config/benchmark/astar_vs_dijkstra.yaml) for the canonical A*-vs-Dijkstra sweep.
