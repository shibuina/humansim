from __future__ import annotations

from collections.abc import Callable

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim.utils.types import (
    BeliefState,
    CommandType,
    HighLevelCommand,
    InteractionContract,
    InteractionState,
    NeedsState,
    NeedState,
    Pose2D,
    WaypointMovement,
)
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentViz as AgentVizMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import AddWalls, SpawnAgents
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Pose2D as RosPose2D
from rclpy.executors import SingleThreadedExecutor

from ._capture import STREAM, settle, spin_until, subscribe

LEVEL2_FIELDS = (
    "vision_range",
    "vision_fov",
    "proximity_sense",
    "observed_agent_id",
    "path_agent_id",
    "path_offset",
    "igoal_agent_id",
    "goal_agent_id",
    "wp_agent_id",
    "wp_offset",
    "wp_active",
)


def _manager(manager_factory: Callable[..., AgentManager], name: str) -> AgentManager:
    scenario = ScenarioConfig(name=name, simulation=SimulationParams(seed=1, dt=0.05, max_ticks=0), modules=ModuleConfig())
    return manager_factory(scenario, node_name=f"test_{name}")


def _spawn(mgr: AgentManager, x: float, kind: int = AgentStateMsg.KIND_HUMAN, goal: tuple[float, float] | None = None) -> int:
    req = SpawnAgents.Request()
    msg = AgentStateMsg()
    msg.kind = kind
    msg.pose = RosPose2D(x=x, y=0.0, theta=0.0)
    msg.desired_velocity = 1.3
    msg.radius = 0.3
    msg.agent_type = "robot" if kind == AgentStateMsg.KIND_ROBOT else "adult"
    if goal is not None:
        wp = WaypointMsg()
        wp.pose = RosPose2D(x=goal[0], y=goal[1], theta=0.5)
        msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return resp.spawned_ids[0]


def _scene(mgr: AgentManager) -> tuple[int, int, int, dict[int, InteractionState]]:
    ped_a = _spawn(mgr, 0.0, goal=(5.0, 0.0))
    ped_b = _spawn(mgr, 2.0, goal=(5.0, 2.0))
    bot = _spawn(mgr, 4.0, kind=AgentStateMsg.KIND_ROBOT)
    mgr._agents[ped_a].needs = NeedsState(needs={"hunger": NeedState(value=40.0), "rest": NeedState(value=75.0)})
    mgr._agents[ped_b].needs = NeedsState(needs={"rest": NeedState(value=10.0)})
    mgr._high_level_cmds[bot] = HighLevelCommand(agent_id=bot, type=CommandType.STOP, target_pose=Pose2D(x=4.0, y=1.0, theta=1.0))
    interactions = {
        7: InteractionState(id=7, type=InteractionType.TALK_TO, participants=[ped_a, bot, 999]),
        9: InteractionState(id=9, type=InteractionType.QUEUE_USE, contract=InteractionContract(queue=[ped_a])),
    }
    return ped_a, ped_b, bot, interactions


def _agents(mgr: AgentManager) -> list:
    return [mgr._agents[aid] for aid in mgr._pool_agent_ids]


def test_agent_viz_level1_carries_agents_labels_needs_interactions(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _manager(manager_factory, "agent_viz_level1")
    ped_a, ped_b, bot, interactions = _scene(mgr)
    mgr._set_marker_level(1)

    msg = mgr._build_agent_viz(_agents(mgr), interactions)

    assert msg.level == 1
    assert msg.header.frame_id == "map"
    assert list(msg.agent_id) == [ped_a, ped_b, bot]
    assert list(msg.x) == [0.0, 2.0, 4.0]
    assert list(msg.radius) == [mgr._agents[aid].params.agent_radius for aid in (ped_a, ped_b, bot)]
    assert list(msg.kind) == [AgentStateMsg.KIND_HUMAN, AgentStateMsg.KIND_HUMAN, AgentStateMsg.KIND_ROBOT]

    assert list(msg.cmd_labels) == ["NAVIGATE", "STOP", "SEEK", "INTR"]
    labels = {aid: msg.cmd_labels[idx] for aid, idx in zip(msg.cmd_agent_id, msg.cmd_label_idx, strict=True)}
    assert labels == {ped_a: "INTR", ped_b: "NAVIGATE", bot: "STOP"}

    needs = [(aid, msg.need_names[idx], slot, value) for aid, idx, slot, value in zip(msg.need_agent_id, msg.need_name_idx, msg.need_slot, msg.need_value, strict=True)]
    assert needs == [(ped_a, "hunger", 0, 40.0), (ped_a, "rest", 1, 75.0), (ped_b, "rest", 0, 10.0)]

    assert list(msg.interaction_id) == [7, 9]
    assert list(msg.interaction_label) == ["TALK [3p]", "QUEUE [0p +1q]"]
    assert list(msg.interaction_offset) == [0, 3, 3]
    assert list(msg.interaction_participants) == [ped_a, bot, 999]

    for field in LEVEL2_FIELDS:
        assert len(getattr(msg, field)) == 0, field


def test_agent_viz_level2_carries_perception_goals_waypoints(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _manager(manager_factory, "agent_viz_level2")
    ped_a, ped_b, bot, interactions = _scene(mgr)
    mgr._set_marker_level(2)
    agents = {aid: mgr._agents[aid] for aid in (ped_a, ped_b, bot)}
    agents[ped_a].belief = BeliefState(agent_id=ped_a, observed_agents=[agents[ped_b].state, agents[bot].state])
    mgr._cached_intermediate_goals = {ped_a: Pose2D(x=1.5, y=0.5, theta=0.0)}
    agents[ped_a].movement = WaypointMovement(waypoints=[Pose2D(x=5.0), Pose2D(x=5.0, y=5.0), Pose2D(y=5.0)], radii=[0.2, 0.0, 0.4], index=2)
    agents[ped_b].movement = WaypointMovement(waypoints=[Pose2D(x=1.0, y=1.0), Pose2D(x=2.0, y=2.0)], radii=[0.5, 0.0], index=1)
    agents[bot].movement = WaypointMovement(waypoints=[Pose2D(x=4.0, y=4.0)])

    msg = mgr._build_agent_viz(_agents(mgr), interactions)

    assert msg.level == 2
    for aid, rng, fov, prox in zip(msg.agent_id, msg.vision_range, msg.vision_fov, msg.proximity_sense, strict=True):
        p = agents[aid].params.perception
        assert (rng, fov, prox) == (p.vision_range, p.vision_fov, p.proximity_sense)

    assert list(msg.observed_agent_id) == [ped_a, ped_a]
    assert list(msg.observed_slot) == [0, 1]
    assert list(msg.observed_x) == [2.0, 4.0]

    assert list(msg.igoal_agent_id) == [ped_a]
    assert (msg.igoal_x[0], msg.igoal_y[0]) == (1.5, 0.5)
    goals = {aid: (x, y, th) for aid, x, y, th in zip(msg.goal_agent_id, msg.goal_x, msg.goal_y, msg.goal_theta, strict=True)}
    assert goals == {ped_a: (5.0, 0.0, 0.5), ped_b: (5.0, 2.0, 0.5), bot: (4.0, 1.0, 1.0)}

    assert list(msg.wp_agent_id) == [ped_a, ped_b, bot]
    assert list(msg.wp_offset) == [0, 3, 5, 6]
    assert list(msg.wp_x) == [5.0, 5.0, 0.0, 1.0, 2.0, 4.0]
    assert list(msg.wp_y) == [0.0, 5.0, 5.0, 1.0, 2.0, 4.0]
    assert list(msg.wp_active) == [2, 1, 0]
    assert list(msg.wp_active_radius) == [0.4, 0.0, 0.3]


def test_agent_viz_paths_follow_global_planner_cache(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _manager(manager_factory, "agent_viz_paths")
    walls = AddWalls.Request()
    for name, (x1, y1), (x2, y2) in (("s", (-10, -10), (10, -10)), ("e", (10, -10), (10, 10)), ("n", (10, 10), (-10, 10)), ("w", (-10, 10), (-10, -10)), ("mid", (3, -3), (3, 3))):
        walls.names.append(name)
        walls.starts.append(Point32(x=float(x1), y=float(y1)))
        walls.ends.append(Point32(x=float(x2), y=float(y2)))
    mgr._add_walls_callback(walls, AddWalls.Response())
    ped = _spawn(mgr, 0.0, goal=(6.0, 0.0))
    mgr._set_marker_level(2)
    mgr.tick()

    msg = mgr._build_agent_viz(_agents(mgr), {})

    path = mgr._agents[ped].global_planner.get_cached_paths()[ped]
    assert len(path) > 1
    assert list(msg.path_agent_id) == [ped]
    assert list(msg.path_offset) == [0, len(path)]
    assert list(msg.path_x) == [wp.x for wp in path]
    assert list(msg.path_y) == [wp.y for wp in path]


def test_marker_level_zero_publishes_one_clearing_viz(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _manager(manager_factory, "agent_viz_clear")
    received = subscribe(mgr, AgentVizMsg, "viz_state", STREAM)
    executor = SingleThreadedExecutor()
    executor.add_node(mgr)
    settle(executor)
    _spawn(mgr, 0.0, goal=(5.0, 0.0))
    _spawn(mgr, 2.0, goal=(5.0, 2.0))

    mgr.tick()
    settle(executor)
    assert received == []

    mgr._set_marker_level(2)
    mgr.tick()
    mgr.tick()
    spin_until(executor, lambda: len(received) == 2)
    assert [(m.level, len(m.agent_id)) for m in received] == [(2, 2), (2, 2)]

    mgr._set_marker_level(0)
    mgr.tick()
    spin_until(executor, lambda: len(received) == 3)
    clear = received[2]
    assert clear.level == 0
    for field in AgentVizMsg.get_fields_and_field_types():
        if field not in ("header", "level"):
            assert len(getattr(clear, field)) == 0, field

    mgr.tick()
    settle(executor)
    assert len(received) == 3
    executor.remove_node(mgr)
