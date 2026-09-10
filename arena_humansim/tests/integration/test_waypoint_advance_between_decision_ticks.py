"""A walker that reaches a waypoint advances to the next one exactly once, however many
plain ticks pass before the next decision tick re-evaluates its arrival latch."""

from __future__ import annotations

import math
from collections.abc import Callable

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3


def _spawn(mgr: AgentManager, points: list[tuple[float, float]], mode: int) -> int:
    msg = AgentStateMsg()
    msg.agent_id = 0
    msg.pose = RosPose2D(x=0.0, y=0.0, theta=0.0)
    msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
    msg.desired_velocity = 1.2
    msg.agent_type = "adult"
    pts = []
    for x, y in points:
        wp = WaypointMsg()
        wp.pose = RosPose2D(x=x, y=y, theta=0.0)
        pts.append(wp)
    msg.waypoints = WaypointsMsg(points=pts, mode=mode)
    req = SpawnAgents.Request()
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return int(resp.spawned_ids[0])


def test_a_repeat_walker_does_not_cycle_its_route_while_latched(manager_factory: Callable[..., AgentManager]) -> None:
    # five waypoints and a decision tick every five ticks: the failure mode had the index
    # back on the reached waypoint at every decision tick, latched for good
    scenario = ScenarioConfig(name="advance", simulation=SimulationParams(seed=0, dt=0.05, max_ticks=10, bt_tick_interval=5), modules=ModuleConfig(global_planner="astar"))
    mgr = manager_factory(scenario, node_name="test_waypoint_advance")
    aid = _spawn(mgr, [(0.1, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0), (0.0, 4.0)], WaypointsMsg.MODE_REPEAT)
    indices: list[int] = []
    farthest = 0.0
    for _ in range(400):  # 20 s
        mgr.tick()
        idx = mgr._pool.idx(aid)
        farthest = max(farthest, math.hypot(float(mgr._pool.pos[idx, 0]), float(mgr._pool.pos[idx, 1])))
        mv = mgr._agents[aid].movement
        if not indices or indices[-1] != mv.index:
            indices.append(mv.index)
    assert farthest > 5.0, f"walked {farthest:.2f} m at most; index history {indices}"
    assert indices == sorted(indices) and len(indices) <= 4, f"index history {indices}: each waypoint is left once"
