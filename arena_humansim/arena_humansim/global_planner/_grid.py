from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from arena_humansim.utils.types import Pose2D, Segments

#: Extra room a spawn snap leaves between an agent's body and the furnished grid.
SPAWN_MARGIN_M = 0.1

#: Free regions smaller than this share of the largest one are pockets the inflation has
#: sealed off; they count as occupied.
POCKET_FRACTION = 0.25


def fill_pockets(grid: np.ndarray, fraction: float = POCKET_FRACTION) -> tuple[np.ndarray, int]:
    """`grid` with every free component smaller than `fraction` of the largest marked
    occupied (value 2), and the number of pockets filled."""
    labels, count = label(grid == 0)
    if count < 2:
        return grid, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    small = sizes < fraction * sizes.max()
    small[0] = False
    pockets = small[labels]
    if not pockets.any():
        return grid, 0
    out = grid.copy()
    out[pockets] = 2
    return out, int(small.sum())


def nearest_free_map(grid: np.ndarray) -> np.ndarray | None:
    """For every cell, the (row, col) of the nearest free cell - `None` when nothing is free.

    One distance transform per grid replaces a ring search, which is pure Python and cost a
    stuck agent (start deep inside a furniture inflation) 100 ms of every tick.
    """
    if not (grid == 0).any():
        return None
    return distance_transform_edt(grid != 0, return_distances=False, return_indices=True)


#: Cells a walk-based snap explores before giving up and taking the straight-line answer.
WALK_LIMIT_CELLS = 40000


def nearest_by_walk(passable: np.ndarray, wanted: np.ndarray, start: tuple[int, int], limit: int = WALK_LIMIT_CELLS) -> tuple[int, int] | None:
    """The first `wanted` cell a 4-connected walk over `passable` cells reaches from `start`
    (which may itself be impassable). `None` when nothing is reached within `limit` cells."""
    rows, cols = passable.shape
    seen = np.zeros_like(passable, dtype=np.bool_)
    seen[start] = True
    queue: deque[tuple[int, int]] = deque([start])
    explored = 0
    while queue and explored < limit:
        r, c = queue.popleft()
        explored += 1
        if wanted[r, c]:
            return (r, c)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and passable[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return None


class SnapMaps:
    """Per-grid lookups behind `snap_pose`: (blocked, nearest) maps per clearance and the
    walk results per (clearance, cell). Replace it with the grid."""

    def __init__(self) -> None:
        self.maps: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
        self.walked: dict[tuple[int, int, int], tuple[int, int] | None] = {}


def snap_pose(
    grid: np.ndarray | None,
    origin: Pose2D,
    resolution: float,
    inflation: float,
    pose: Pose2D,
    radius: float,
    cache: SnapMaps,
    passable: np.ndarray | None = None,
) -> Pose2D:
    """`pose` when an agent of `radius` fits there, else the nearest cell where it does.

    A cell fits when it is free and at least `radius + SPAWN_MARGIN_M - inflation` from the
    occupied region (`radius` 0 asks for any free cell, as a waypoint does). With `passable`
    (the walls alone, uninflated) the nearest cell is the first one a walk from `pose` reaches
    without crossing a wall; without it, the nearest by straight line - which is on the far
    side of a wall as often as not when the room's whole free space is a sealed pocket.
    """
    if grid is None:
        return pose
    rows, cols = grid.shape
    rc = world_to_grid(origin, resolution, pose.x, pose.y)
    if not (0 <= rc[0] < rows and 0 <= rc[1] < cols):
        return pose
    k = int(math.ceil(max(radius + SPAWN_MARGIN_M - inflation, 0.0) / resolution)) if radius > 0 else 0
    maps = cache.maps.get(k)
    if maps is None:
        blocked = grid != 0
        if k > 0:
            blocked = blocked | (distance_transform_edt(grid == 0) < k)
        blocked = blocked.astype(np.uint8)
        maps = (blocked, nearest_free_map(blocked))
        cache.maps[k] = maps
    blocked, nearest = maps
    if not blocked[rc[0], rc[1]] or nearest is None:
        return pose
    snapped: tuple[int, int] | None = None
    if passable is not None:
        key = (k, rc[0], rc[1])
        if key not in cache.walked:
            cache.walked[key] = nearest_by_walk(passable == 0, blocked == 0, rc)
        snapped = cache.walked[key]
    if snapped is None:
        snapped = (int(nearest[0, rc[0], rc[1]]), int(nearest[1, rc[0], rc[1]]))
    if snapped == rc:
        return pose
    cell = grid_to_world(origin, resolution, snapped[0], snapped[1])
    return Pose2D(x=cell.x, y=cell.y, theta=pose.theta)


def world_to_grid(origin: Pose2D, resolution: float, wx: float, wy: float) -> tuple[int, int]:
    col = int(round((wx - origin.x) / resolution))
    row = int(round((wy - origin.y) / resolution))
    return row, col


def grid_to_world(origin: Pose2D, resolution: float, row: int, col: int) -> Pose2D:
    return Pose2D(x=col * resolution + origin.x, y=row * resolution + origin.y)


def line_of_sight(grid: np.ndarray, origin: Pose2D, resolution: float, p1: Pose2D, p2: Pose2D) -> bool:
    r1, c1 = world_to_grid(origin, resolution, p1.x, p1.y)
    r2, c2 = world_to_grid(origin, resolution, p2.x, p2.y)
    rows, cols = grid.shape
    # Manhattan step count so thin diagonal walls aren't skipped.
    steps = abs(r2 - r1) + abs(c2 - c1)
    if steps == 0:
        return True
    for i in range(steps + 1):
        t = i / steps
        r = int(round(r1 + t * (r2 - r1)))
        c = int(round(c1 + t * (c2 - c1)))
        # Out-of-grid cells are free: the grid only spans walls + margin.
        if 0 <= r < rows and 0 <= c < cols and grid[r, c] != 0:
            return False
    return True


def simplify_with_los(
    grid: np.ndarray,
    origin: Pose2D,
    resolution: float,
    waypoints: list[Pose2D],
) -> list[Pose2D]:
    if len(waypoints) <= 2:
        return waypoints
    result = [waypoints[0]]
    i = 0
    while i < len(waypoints) - 1:
        farthest = i + 1
        for j in range(len(waypoints) - 1, i + 1, -1):
            if line_of_sight(grid, origin, resolution, waypoints[i], waypoints[j]):
                farthest = j
                break
        result.append(waypoints[farthest])
        i = farthest
    return result


def push_from_walls(
    wall_segments: Segments,
    inflation_radius: float,
    waypoints: list[Pose2D],
) -> list[Pose2D]:
    if len(waypoints) <= 2:
        return waypoints
    margin = inflation_radius
    margin_sq = margin * margin
    result = [waypoints[0]]
    for wp in waypoints[1:-1]:
        px, py = wp.x, wp.y
        push_x, push_y = 0.0, 0.0
        for (x1, y1), (x2, y2) in wall_segments:
            sx, sy = x2 - x1, y2 - y1
            seg_len_sq = sx * sx + sy * sy
            if seg_len_sq < 1e-12:
                cx, cy = x1, y1
            else:
                t = max(0.0, min(1.0, ((px - x1) * sx + (py - y1) * sy) / seg_len_sq))
                cx, cy = x1 + t * sx, y1 + t * sy
            dx, dy = px - cx, py - cy
            dist_sq = dx * dx + dy * dy
            if dist_sq < margin_sq and dist_sq > 1e-12:
                dist = math.sqrt(dist_sq)
                nx, ny = dx / dist, dy / dist
                push_x += nx * (margin - dist)
                push_y += ny * (margin - dist)
        result.append(Pose2D(x=px + push_x, y=py + push_y))
    result.append(waypoints[-1])
    return result


def min_distance_to_path(pos: Pose2D, waypoints: Iterable[Pose2D]) -> float:
    best = math.inf
    for wp in waypoints:
        d = math.hypot(pos.x - wp.x, pos.y - wp.y)
        if d < best:
            best = d
    return best


def needs_replan(
    path_cache: dict[int, tuple[tuple[float, float], list[Pose2D], int]],
    agent_id: int,
    goal: Pose2D,
    agent_pos: Pose2D,
    replan_distance: float,
) -> bool:
    if agent_id not in path_cache:
        return True
    cached_goal, waypoints, _ = path_cache[agent_id]
    goal_key = (round(goal.x, 3), round(goal.y, 3))
    if cached_goal != goal_key:
        return True
    if not waypoints:
        return True
    return min_distance_to_path(agent_pos, waypoints) > replan_distance


def next_waypoint(waypoints: Sequence[Pose2D], idx: int) -> Pose2D:
    target = idx + 1
    if target < len(waypoints):
        return waypoints[target]
    return waypoints[-1]
