from __future__ import annotations

import pytest
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import Waypoints
from arena_humansim_msgs.srv import SpawnAgents as SpawnAgentsSrv
from geometry_msgs.msg import Pose2D as Pose2DMsg
from geometry_msgs.msg import Vector3

from arena_humansim.core.pool import KIND_ROBOT
from tests.ros._helpers import (
    RemoveAgents,
    RosTestSystem,
    SpawnAgents,
    make_remove_request,
    make_spawn_request,
)

pytestmark = pytest.mark.ros

_EXT_ID = 424242


@pytest.fixture(scope="module")
def system(ros_system: RosTestSystem) -> RosTestSystem:
    return ros_system


@pytest.fixture(autouse=True)
def _clean(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    system.manager._latest_world_state = None


def _world_state(entries: list[tuple[int, float, float, float, float]], stamp_ms: int = 0) -> AgentStatesMsg:
    msg = AgentStatesMsg()
    msg.header.frame_id = "map"
    msg.header.stamp.sec = stamp_ms // 1000
    msg.header.stamp.nanosec = stamp_ms % 1000 * 1_000_000
    for aid, x, y, theta, radius in entries:
        a = AgentStateMsg()
        a.agent_id = aid
        a.pose = Pose2DMsg(x=x, y=y, theta=theta)
        a.radius = radius
        msg.agents.append(a)
    return msg


def _spawn_robot(system: RosTestSystem, x: float, y: float) -> int:
    msg = AgentStateMsg()
    msg.agent_id = 0
    msg.pose = Pose2DMsg(x=x, y=y, theta=0.0)
    msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
    msg.desired_velocity = 1.0
    msg.radius = 0.3
    msg.agent_type = "guard_bot"
    msg.kind = AgentStateMsg.KIND_ROBOT
    msg.waypoints = Waypoints()
    req = SpawnAgentsSrv.Request()
    req.agents.append(msg)
    resp = system.call(SpawnAgentsSrv, "spawn_agents", req)
    assert resp.success is True
    return resp.spawned_ids[0]


def test_world_state_spawns_external_entity(system: RosTestSystem) -> None:
    manager = system.manager
    manager._world_state_callback(_world_state([(_EXT_ID, 1.0, 2.0, 0.5, 0.4)]))
    system.tick_manager(1)

    entity = manager._external_entities[_EXT_ID]
    pool = manager._pool
    idx = pool._id_to_idx[entity.agent_id]
    assert entity.owned is True
    assert int(pool.kind[idx]) == KIND_ROBOT
    assert int(pool.policy_idx[idx]) == -1
    assert pool.pos[idx, 0] == pytest.approx(1.0)
    assert pool.pos[idx, 1] == pytest.approx(2.0)
    assert pool.theta[idx] == pytest.approx(0.5)
    assert pool.agent_radius[idx] == pytest.approx(0.4)


def test_world_state_teleports_existing_entity(system: RosTestSystem) -> None:
    manager = system.manager
    manager._world_state_callback(_world_state([(_EXT_ID, 1.0, 0.0, 0.0, 0.4)]))
    system.tick_manager(1)
    aid = manager._external_entities[_EXT_ID].agent_id

    manager._world_state_callback(_world_state([(_EXT_ID, 2.0, 0.5, 1.0, 0.4)]))
    system.tick_manager(1)

    pool = manager._pool
    assert manager._external_entities[_EXT_ID].agent_id == aid
    assert pool.n == 1
    idx = pool._id_to_idx[aid]
    assert pool.pos[idx, 0] == pytest.approx(2.0)
    assert pool.pos[idx, 1] == pytest.approx(0.5)
    assert pool.theta[idx] == pytest.approx(1.0)


def test_neighbors_perceive_external_entity(system: RosTestSystem) -> None:
    manager = system.manager
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request([{"x": 0.0, "y": 0.0}]))
    ped_aid = resp.spawned_ids[0]

    manager._world_state_callback(_world_state([(_EXT_ID, 1.2, 0.0, 0.0, 0.4)]))
    system.tick_manager(1)

    ext_aid = manager._external_entities[_EXT_ID].agent_id
    assert ext_aid in manager._pool.visible_agent_ids(ped_aid)


def test_world_state_adopts_registered_robot(system: RosTestSystem) -> None:
    manager = system.manager
    robot_aid = _spawn_robot(system, 3.0, 0.0)
    assert manager._pool.n == 1

    manager._world_state_callback(_world_state([(_EXT_ID, 3.05, 0.0, 0.2, 0.5)]))
    system.tick_manager(1)

    entity = manager._external_entities[_EXT_ID]
    pool = manager._pool
    assert entity.agent_id == robot_aid
    assert entity.owned is False
    assert pool.n == 1
    idx = pool._id_to_idx[robot_aid]
    assert pool.pos[idx, 0] == pytest.approx(3.05)
    assert pool.agent_radius[idx] == pytest.approx(0.5)


def test_external_entity_expires(system: RosTestSystem) -> None:
    manager = system.manager
    manager._world_state_callback(_world_state([(_EXT_ID, 1.0, 0.0, 0.0, 0.4)]))
    system.tick_manager(1)
    aid = manager._external_entities[_EXT_ID].agent_id
    assert aid in manager._agents

    system.tick_manager(manager._external_timeout_ticks + 2)

    assert _EXT_ID not in manager._external_entities
    assert aid not in manager._agents
    assert manager._pool.n == 0


def test_mirror_carries_velocity_between_stamped_poses(system: RosTestSystem) -> None:
    manager = system.manager
    manager._world_state_callback(_world_state([(_EXT_ID, 1.0, 0.0, 0.0, 0.4)], stamp_ms=1000))
    system.tick_manager(1)
    manager._world_state_callback(_world_state([(_EXT_ID, 1.05, -0.02, 0.0, 0.4)], stamp_ms=1050))
    system.tick_manager(1)

    aid = manager._external_entities[_EXT_ID].agent_id
    pool = manager._pool
    idx = pool._id_to_idx[aid]
    assert pool.vel[idx, 0] == pytest.approx(1.0)
    assert pool.vel[idx, 1] == pytest.approx(-0.4)
    assert manager._agents[aid].state.velocity == pytest.approx((1.0, -0.4))


def test_adopted_robot_carries_velocity_between_stamped_poses(system: RosTestSystem) -> None:
    manager = system.manager
    robot_aid = _spawn_robot(system, 3.0, 0.0)
    manager._world_state_callback(_world_state([(_EXT_ID, 3.0, 0.0, 0.0, 0.5)], stamp_ms=1000))
    system.tick_manager(1)
    manager._world_state_callback(_world_state([(_EXT_ID, 3.0, 0.04, 0.0, 0.5)], stamp_ms=1050))
    system.tick_manager(1)

    pool = manager._pool
    idx = pool._id_to_idx[robot_aid]
    assert pool.vel[idx, 0] == pytest.approx(0.0)
    assert pool.vel[idx, 1] == pytest.approx(0.8)


def test_mirror_velocity_lapses_when_the_feed_stops(system: RosTestSystem) -> None:
    manager = system.manager
    manager._world_state_callback(_world_state([(_EXT_ID, 1.0, 0.0, 0.0, 0.4)], stamp_ms=1000))
    system.tick_manager(1)
    manager._world_state_callback(_world_state([(_EXT_ID, 1.05, 0.0, 0.0, 0.4)], stamp_ms=1050))
    system.tick_manager(1)
    system.tick_manager(manager._external_velocity_ticks + 1)

    pool = manager._pool
    idx = pool._id_to_idx[manager._external_entities[_EXT_ID].agent_id]
    assert pool.vel[idx, 0] == pytest.approx(0.0)
    assert pool.pos[idx, 0] == pytest.approx(1.05 + 0.05 * (manager._external_velocity_ticks + 1))


@pytest.mark.parametrize(
    ("x", "stamp_ms"),
    [(6.0, 1050), (1.5, 1500), (1.05, 1000)],
    ids=["jump", "stale_gap", "repeated_stamp"],
)
def test_mirror_carries_no_velocity_across(system: RosTestSystem, x: float, stamp_ms: int) -> None:
    manager = system.manager
    manager._world_state_callback(_world_state([(_EXT_ID, 1.0, 0.0, 0.0, 0.4)], stamp_ms=1000))
    system.tick_manager(1)
    manager._world_state_callback(_world_state([(_EXT_ID, x, 0.0, 0.0, 0.4)], stamp_ms=stamp_ms))
    system.tick_manager(1)

    pool = manager._pool
    idx = pool._id_to_idx[manager._external_entities[_EXT_ID].agent_id]
    assert pool.vel[idx, 0] == pytest.approx(0.0)
    assert pool.pos[idx, 0] == pytest.approx(x)


def test_mirror_carries_a_fed_velocity(system: RosTestSystem) -> None:
    manager = system.manager
    msg = _world_state([(_EXT_ID, 1.0, 0.0, 0.0, 0.4)])
    msg.agents[0].velocity = Vector3(x=0.6, y=0.3, z=0.0)
    manager._world_state_callback(msg)
    system.tick_manager(1)

    pool = manager._pool
    idx = pool._id_to_idx[manager._external_entities[_EXT_ID].agent_id]
    assert pool.vel[idx, 0] == pytest.approx(0.6)
    assert pool.vel[idx, 1] == pytest.approx(0.3)


def test_external_entities_excluded_from_agent_states(system: RosTestSystem) -> None:
    manager = system.manager
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request([{"x": 0.0, "y": 0.0}]))
    ped_aid = resp.spawned_ids[0]

    manager._world_state_callback(_world_state([(_EXT_ID, 1.0, 0.0, 0.0, 0.4)]))
    system.tick_manager(1)

    ext_aid = manager._external_entities[_EXT_ID].agent_id
    published = set(manager._build_agent_frame().agent_id)
    assert ped_aid in published
    assert ext_aid not in published


def test_world_state_binds_internal_agent(system: RosTestSystem) -> None:
    manager = system.manager
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request([{"x": 0.0, "y": 0.0}]))
    ped_aid = resp.spawned_ids[0]
    pool = manager._pool
    saved = int(pool.policy_idx[pool._id_to_idx[ped_aid]])
    assert saved != -1

    manager._world_state_callback(_world_state([(ped_aid, 4.0, 5.0, 0.7, 0.3)]))
    system.tick_manager(1)

    entity = manager._external_entities[ped_aid]
    assert entity.agent_id == ped_aid
    assert entity.owned is False
    assert entity.saved_policy_idx == saved
    assert pool.n == 1
    idx = pool._id_to_idx[ped_aid]
    assert int(pool.policy_idx[idx]) == -1
    assert pool.pos[idx, 0] == pytest.approx(4.0)
    assert pool.pos[idx, 1] == pytest.approx(5.0)
    assert pool.theta[idx] == pytest.approx(0.7)


def test_bound_agent_renewal_keeps_pin(system: RosTestSystem) -> None:
    manager = system.manager
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request([{"x": 0.0, "y": 0.0}]))
    ped_aid = resp.spawned_ids[0]
    pool = manager._pool
    saved = int(pool.policy_idx[pool._id_to_idx[ped_aid]])

    manager._world_state_callback(_world_state([(ped_aid, 1.0, 1.0, 0.0, 0.3)]))
    system.tick_manager(1)
    manager._world_state_callback(_world_state([(ped_aid, 2.0, 1.5, 0.4, 0.3)]))
    system.tick_manager(1)

    entity = manager._external_entities[ped_aid]
    assert entity.agent_id == ped_aid
    assert entity.saved_policy_idx == saved
    assert pool.n == 1
    idx = pool._id_to_idx[ped_aid]
    assert int(pool.policy_idx[idx]) == -1
    assert pool.pos[idx, 0] == pytest.approx(2.0)
    assert pool.pos[idx, 1] == pytest.approx(1.5)


def test_bound_agent_carries_fed_velocity(system: RosTestSystem) -> None:
    manager = system.manager
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request([{"x": 0.0, "y": 0.0}]))
    ped_aid = resp.spawned_ids[0]
    pool = manager._pool

    msg = _world_state([(ped_aid, 4.0, 5.0, 0.0, 0.3)])
    msg.agents[0].velocity = Vector3(x=0.8, y=-0.2, z=0.0)
    manager._world_state_callback(msg)
    system.tick_manager(1)

    idx = pool._id_to_idx[ped_aid]
    assert int(pool.policy_idx[idx]) == -1
    assert pool.vel[idx, 0] == pytest.approx(0.8)
    assert pool.vel[idx, 1] == pytest.approx(-0.2)

    system.tick_manager(1)

    idx = pool._id_to_idx[ped_aid]
    assert pool.vel[idx, 0] == pytest.approx(0.8)
    assert pool.vel[idx, 1] == pytest.approx(-0.2)


def test_bound_agent_expiry_restores_autonomy(system: RosTestSystem) -> None:
    manager = system.manager
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request([{"x": 0.0, "y": 0.0}]))
    ped_aid = resp.spawned_ids[0]
    pool = manager._pool
    saved = int(pool.policy_idx[pool._id_to_idx[ped_aid]])

    manager._world_state_callback(_world_state([(ped_aid, 4.0, 5.0, 0.0, 0.3)]))
    system.tick_manager(1)
    system.tick_manager(manager._external_timeout_ticks + 2)

    assert ped_aid not in manager._external_entities
    assert ped_aid in manager._agents
    assert pool.n == 1
    assert int(pool.policy_idx[pool._id_to_idx[ped_aid]]) == saved


def test_bound_agent_included_in_agent_states(system: RosTestSystem) -> None:
    manager = system.manager
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request([{"x": 0.0, "y": 0.0}]))
    ped_aid = resp.spawned_ids[0]

    manager._world_state_callback(_world_state([(ped_aid, 1.0, 0.0, 0.0, 0.3)]))
    system.tick_manager(1)

    assert manager._external_entities[ped_aid].agent_id == ped_aid
    published = set(manager._build_agent_frame().agent_id)
    assert ped_aid in published


def test_ghost_spawn_unchanged_alongside_bound_agent(system: RosTestSystem) -> None:
    manager = system.manager
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request([{"x": 0.0, "y": 0.0}]))
    ped_aid = resp.spawned_ids[0]

    manager._world_state_callback(_world_state([(ped_aid, 1.0, 0.0, 0.0, 0.3), (_EXT_ID, 8.0, 8.0, 0.0, 0.4)]))
    system.tick_manager(1)

    ghost = manager._external_entities[_EXT_ID]
    assert ghost.owned is True
    assert ghost.saved_policy_idx is None
    assert ghost.agent_id != ped_aid
    assert manager._pool.n == 2
    published = set(manager._build_agent_frame().agent_id)
    assert ped_aid in published
    assert ghost.agent_id not in published
