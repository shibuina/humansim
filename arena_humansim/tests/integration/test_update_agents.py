from __future__ import annotations

from collections.abc import Callable

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import SpawnAgents, UpdateAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3


def _spawn_one(mgr: AgentManager, x: float = 0.0) -> int:
    req = SpawnAgents.Request()
    msg = AgentStateMsg()
    msg.agent_id = 0
    msg.pose = RosPose2D(x=x, y=0.0, theta=0.0)
    msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
    msg.desired_velocity = 1.3
    msg.agent_type = "adult"
    wp = WaypointMsg()
    wp.pose = RosPose2D(x=x + 10.0, y=0.0, theta=0.0)
    msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return int(resp.spawned_ids[0])


def _update(mgr: AgentManager, **fields: object) -> UpdateAgents.Response:
    req = UpdateAgents.Request()
    msg = AgentStateMsg()
    for k, v in fields.items():
        setattr(msg, k, v)
    req.agents.append(msg)
    resp = UpdateAgents.Response()
    mgr._update_agents_callback(req, resp)
    return resp


def test_update_changes_parameters_in_place(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_update_agents")
    aid = _spawn_one(mgr)
    idx = mgr._pool.idx(aid)
    mgr._pool.pos[idx] = (3.0, 4.0)
    before_route = list(mgr._agents[aid].movement.waypoints)

    resp = _update(mgr, agent_id=aid, desired_velocity=0.6, radius=0.4, vision_range=2.0, max_velocity=2.6, repulsion_strength=0.5)
    assert resp.success and list(resp.updated_ids) == [aid]

    agent = mgr._agents[aid]
    assert agent.state.desired_velocity == 0.6 and agent.params.agent_radius == 0.4
    assert agent.params.perception.vision_range == 2.0 and agent.params.max_velocity == 2.6
    assert agent.params.local_planner_params["repulsion_strength"] == 0.5
    pool = mgr._pool
    assert pool.desired_vel[idx] == 0.6 and pool.agent_radius[idx] == 0.4
    assert pool.vision_range[idx] == 2.0 and pool.max_velocity[idx] == 2.6
    assert tuple(pool.pos[idx]) == (3.0, 4.0), "an update must not move the agent"
    assert list(agent.movement.waypoints) == before_route, "an update must not touch the route"
    sfm = mgr._local_planner
    if hasattr(sfm, "_repulsion_strength"):
        assert sfm._repulsion_strength[idx] == 0.5

    # an unknown id is reported, not fatal; a type re-resolves the whole set (one manager per
    # process: the factory cannot change the logger severity twice)
    resp = _update(mgr, agent_id=aid + 99, desired_velocity=0.6)
    assert not resp.success and "unknown" in resp.message
    resp = _update(mgr, agent_id=aid, agent_type="elder")
    assert resp.success and mgr._agents[aid].params.name == "elder"
    assert tuple(pool.pos[idx]) == (3.0, 4.0)
