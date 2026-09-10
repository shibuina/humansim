"""A walker whose first waypoint is close by (the base scenarios write `pose: <room>` and
`waypoints: [<room>, ...]`, two independent points of one room) must not freeze on it."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3


def _spawn(mgr: AgentManager, x: float, first_dx: float, mode: int) -> int:
    req = SpawnAgents.Request()
    msg = AgentStateMsg()
    msg.agent_id = 0
    msg.pose = RosPose2D(x=x, y=0.0, theta=0.0)
    msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
    msg.desired_velocity = 1.2
    msg.agent_type = "adult"
    pts = []
    for wx in (x + first_dx, x + 12.0):
        wp = WaypointMsg()
        wp.pose = RosPose2D(x=wx, y=0.0, theta=0.0)
        pts.append(wp)
    msg.waypoints = WaypointsMsg(points=pts, mode=mode)
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return int(resp.spawned_ids[0])


@pytest.mark.parametrize("first_dx", [0.05, 0.12, 0.7, 2.0])
def test_a_walker_leaves_a_first_waypoint_next_to_its_spawn(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig, first_dx: float) -> None:
    mgr = manager_factory(minimal_scenario, node_name=f"test_first_wp_{int(first_dx * 100)}")
    for mode in (WaypointsMsg.MODE_REPEAT, WaypointsMsg.MODE_REVERSE):
        aid = _spawn(mgr, x=float(mode) * 30.0, first_dx=first_dx, mode=mode)
        idx = mgr._pool.idx(aid)
        start = (float(mgr._pool.pos[idx, 0]), float(mgr._pool.pos[idx, 1]))
        farthest = 0.0
        for _ in range(400):  # 20 s at dt 0.05; a REPEAT shuttle may be back near its start by then
            mgr.tick()
            idx = mgr._pool.idx(aid)
            farthest = max(farthest, math.dist(start, (float(mgr._pool.pos[idx, 0]), float(mgr._pool.pos[idx, 1]))))
        assert farthest > 5.0, f"mode {mode}: got {farthest:.2f} m from its spawn in 20 s with the first waypoint {first_dx} m away (index {mgr._agents[aid].movement.index}, latched {bool(mgr._pool.latched[idx])})"
