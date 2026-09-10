"""A pedestrian whose spawn lands inside a piece of furniture (the scenario resolves a room
name to a point checked against walls only) is moved to the nearest clear spot and walks."""

from __future__ import annotations

import math
from collections.abc import Callable

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import ObstacleConfig
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import AddObstacles, AddWalls, SpawnAgents
from geometry_msgs.msg import Point32, Vector3
from geometry_msgs.msg import Pose2D as RosPose2D


def _furnish(mgr: AgentManager) -> None:
    walls = AddWalls.Request()
    for i, (a, b) in enumerate([((-6.0, -6.0), (6.0, -6.0)), ((6.0, -6.0), (6.0, 6.0)), ((6.0, 6.0), (-6.0, 6.0)), ((-6.0, 6.0), (-6.0, -6.0))]):
        walls.names.append(f"wall_{i}")
        walls.starts.append(Point32(x=a[0], y=a[1], z=0.0))
        walls.ends.append(Point32(x=b[0], y=b[1], z=0.0))
    mgr._add_walls_callback(walls, AddWalls.Response())
    sofa = ObstacleConfig()
    sofa.name = "sofa"
    sofa.pose = RosPose2D(x=0.0, y=0.0, theta=0.0)
    sofa.bb_x_min, sofa.bb_x_max, sofa.bb_y_min, sofa.bb_y_max, sofa.bb_z_min, sofa.bb_z_max = -1.0, 1.0, -0.5, 0.5, 0.0, 0.8
    req = AddObstacles.Request()
    req.obstacles.append(sofa)
    mgr._add_obstacles_callback(req, AddObstacles.Response())


def _spawn(mgr: AgentManager, x: float, y: float, goal: tuple[float, float]) -> int:
    msg = AgentStateMsg()
    msg.agent_id = 0
    msg.pose = RosPose2D(x=x, y=y, theta=0.0)
    msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
    msg.desired_velocity = 1.0
    msg.agent_type = "adult"
    wp = WaypointMsg()
    wp.pose = RosPose2D(x=goal[0], y=goal[1], theta=0.0)
    msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
    req = SpawnAgents.Request()
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return int(resp.spawned_ids[0])


def test_a_spawn_inside_a_sofa_is_moved_out_and_the_agent_walks(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_spawn_inside_furniture")
    _furnish(mgr)
    aid = _spawn(mgr, 0.3, 0.1, goal=(4.0, 4.0))
    idx = mgr._pool.idx(aid)
    start = (float(mgr._pool.pos[idx, 0]), float(mgr._pool.pos[idx, 1]))
    assert not (-1.3 < start[0] < 1.3 and -0.8 < start[1] < 0.8), f"still inside the sofa's inflation at {start}"
    assert math.dist(start, (0.3, 0.1)) < 1.5
    farthest = 0.0
    for _ in range(300):
        mgr.tick()
        idx = mgr._pool.idx(aid)
        farthest = max(farthest, math.dist(start, (float(mgr._pool.pos[idx, 0]), float(mgr._pool.pos[idx, 1]))))
    assert farthest > 3.0, f"walked only {farthest:.2f} m from {start}"


def test_a_clear_spawn_is_not_touched(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_spawn_clear")
    _furnish(mgr)
    aid = _spawn(mgr, -4.0, -4.0, goal=(4.0, 4.0))
    idx = mgr._pool.idx(aid)
    assert (float(mgr._pool.pos[idx, 0]), float(mgr._pool.pos[idx, 1])) == (-4.0, -4.0)
