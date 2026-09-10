import numpy as np
from arena_humansim.core.agent_manager import ObstacleData, global_plan_segments
from arena_humansim.global_planner._grid import world_to_grid
from arena_humansim.global_planner.astar import AStarPlanner
from arena_humansim.utils.types import Pose2D, Segments


def _obs(name: str, w: float, h: float) -> ObstacleData:
    seg = (((0.0, 0.0), (w, 0.0)),)
    return ObstacleData(name=name, pose=Pose2D(), bb=(-w / 2, w / 2, -h / 2, h / 2, 0.0, 1.0), interaction_types=(), obstacle_type="", wall_segments=seg)


def test_small_furniture_is_thin_and_large_furniture_full() -> None:
    walls = [((0.0, 0.0), (5.0, 0.0))]
    chair, table = _obs("chair", 0.42, 0.57), _obs("table", 0.9, 2.0)
    assert global_plan_segments(walls, [chair, table], 0.7) == (walls + list(table.wall_segments), list(chair.wall_segments))
    assert global_plan_segments(walls, [chair, table], 0.0) == (walls + list(chair.wall_segments) + list(table.wall_segments), [])


def _box(x0: float, y0: float, x1: float, y1: float) -> list:
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


def test_walls_only_fallback_routes_through_furniture_that_seals_a_room() -> None:
    import logging

    from arena_humansim.global_planner._grid import world_to_grid
    from arena_humansim.global_planner.astar import _astar_path

    planner = AStarPlanner(inflation_radius=0.2, resolution=0.1)
    planner.__dict__["_logger"] = logging.getLogger("test")
    # a closed corridor 0 < y < 1.2; a table across it seals it in the furnished grid
    walls = [((0.0, 0.0), (6.0, 0.0)), ((0.0, 1.2), (6.0, 1.2)), ((0.0, 0.0), (0.0, 1.2)), ((6.0, 0.0), (6.0, 1.2))]
    table = _box(2.8, 0.0, 3.2, 1.2)
    planner.set_walls(walls + table, fallback_segments=walls)
    a = world_to_grid(planner._origin, 0.1, 1.0, 0.6)
    b = world_to_grid(planner._origin, 0.1, 5.0, 0.6)
    assert _astar_path(planner._weights, planner._occupancy_grid, a, b, planner._nearest) is None
    assert _astar_path(planner._fallback_weights, planner._fallback_grid, a, b, planner._fallback_nearest) is not None


def test_nearest_free_map_matches_the_ring_search() -> None:
    import numpy as np
    from arena_humansim.global_planner.astar import _nearest_free_cell, nearest_free_map

    grid = np.zeros((20, 20), dtype=np.uint8)
    grid[5:15, 5:15] = 1
    near = nearest_free_map(grid)
    for r, c in ((7, 7), (10, 10), (14, 5), (0, 0), (12, 13)):
        slow = _nearest_free_cell(grid, r, c)
        fast = _nearest_free_cell(grid, r, c, nearest=near)
        assert slow is not None and fast is not None
        assert grid[fast] == 0 and abs(fast[0] - r) + abs(fast[1] - c) <= abs(slow[0] - r) + abs(slow[1] - c) + 1
    assert nearest_free_map(np.ones((3, 3), dtype=np.uint8)) is None


def test_thin_segments_block_their_footprint_without_inflation() -> None:
    import logging

    planner = AStarPlanner(inflation_radius=0.3, resolution=0.1)
    planner.__dict__["_logger"] = logging.getLogger("test")
    wall = [((0.0, 0.0), (4.0, 0.0))]
    chair = [((2.0, 1.0), (2.4, 1.0)), ((2.4, 1.0), (2.4, 1.4)), ((2.4, 1.4), (2.0, 1.4)), ((2.0, 1.4), (2.0, 1.0))]
    planner.set_walls(wall, thin_segments=chair)
    grid, origin, res = planner._occupancy_grid, planner._origin, planner._resolution

    def blocked(x: float, y: float) -> bool:
        return bool(grid[int(round((y - origin.y) / res)), int(round((x - origin.x) / res))])

    assert blocked(2.2, 1.0) and blocked(2.0, 1.2)  # the chair's own edges
    assert blocked(2.2, 0.25)  # within the wall's inflation
    assert not blocked(2.2, 1.6)  # 0.2 m off the chair: only the wall is inflated
    assert not blocked(2.2, 0.45)
    assert np.isinf(planner._weights[int(round((1.0 - origin.y) / res)), int(round((2.2 - origin.x) / res))])


def test_pockets_the_inflation_seals_off_count_as_occupied() -> None:
    from arena_humansim.global_planner._grid import fill_pockets

    room: Segments = [((0.0, 0.0), (10.0, 0.0)), ((10.0, 0.0), (10.0, 10.0)), ((10.0, 10.0), (0.0, 10.0)), ((0.0, 10.0), (0.0, 0.0))]
    # two chair rows 0.6 m apart across the room's south end: the strip between them is
    # free at 0.25 m inflation but unreachable
    rows: Segments = []
    for y0 in (2.0, 3.3):
        rows += [((0.0, y0), (10.0, y0)), ((10.0, y0), (10.0, y0 + 0.7)), ((10.0, y0 + 0.7), (0.0, y0 + 0.7)), ((0.0, y0 + 0.7), (0.0, y0))]
    planner = AStarPlanner(inflation_radius=0.25, resolution=0.1)
    planner.set_walls(room + rows)
    grid = planner._occupancy_grid
    assert grid is not None
    inside_strip = planner.snap_terminal(Pose2D(x=5.0, y=3.0, theta=0.0))
    assert inside_strip.y > 4.0, f"a waypoint in the sealed strip is snapped into reachable space, got {inside_strip}"
    south = planner.snap_spawn(Pose2D(x=5.0, y=1.0, theta=0.0), 0.3)
    assert south.y > 4.0, f"a spawn in the sealed south end is moved into reachable space, got {south}"
    assert planner.snap_spawn(Pose2D(x=5.0, y=7.0, theta=0.0), 0.3).y == 7.0
    raw = np.zeros((20, 20), dtype=np.uint8)
    raw[10, :] = 1
    raw[10, 5] = 0  # one-cell gap keeps the halves one component
    assert fill_pockets(raw)[1] == 0
    raw[10, 5] = 1
    filled, count = fill_pockets(raw)
    assert count == 0 and (filled == raw).all(), "two equal halves are not pockets of each other"


def test_snaps_walk_round_walls_instead_of_jumping_through_them() -> None:
    # a corridor above a room whose whole free space is sealed by a chair row; the nearest
    # free cell by straight line lies in the corridor beyond the room's north wall, the walk
    # leaves through the door on the east side
    walls: Segments = [
        ((0.0, 0.0), (10.0, 0.0)),
        ((10.0, 0.0), (10.0, 16.0)),
        ((10.0, 16.0), (0.0, 16.0)),
        ((0.0, 16.0), (0.0, 0.0)),
        ((0.0, 3.0), (8.5, 3.0)),  # partition with a 1.5 m door at x 8.5-10; the hall north of it is the main space
    ]
    row: Segments = [((0.0, 1.4), (10.0, 1.4)), ((10.0, 1.4), (10.0, 2.1)), ((10.0, 2.1), (0.0, 2.1)), ((0.0, 2.1), (0.0, 1.4))]
    cabinet: Segments = [((8.5, 2.6), (10.0, 2.6)), ((10.0, 2.6), (10.0, 3.3)), ((10.0, 3.3), (8.5, 3.3)), ((8.5, 3.3), (8.5, 2.6))]  # blocks the door from the south
    planner = AStarPlanner(inflation_radius=0.25, resolution=0.1)
    planner.set_walls(walls + row + cabinet, fallback_segments=walls)
    inside = Pose2D(x=2.0, y=2.5, theta=0.0)  # between the row and the partition: a pocket
    assert planner._occupancy_grid is not None and planner._occupancy_grid[world_to_grid(planner._origin, 0.1, 2.0, 2.5)] != 0
    for snapped in (planner.snap_terminal(inside), planner.snap_spawn(inside, 0.3)):
        assert snapped.y < 3.0 or snapped.x > 8.5, f"snapped through the partition to {snapped}"
    # without the walls grid the straight-line answer crosses the partition
    planner.set_walls(walls + row + cabinet)
    assert planner.snap_terminal(inside).y > 3.0
