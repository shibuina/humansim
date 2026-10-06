from __future__ import annotations

from collections.abc import Callable

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.utils.scenario import AgentConfig, InteractionScript, ModuleConfig, ScenarioConfig, SimulationParams, WallConfig
from arena_humansim.utils.types import InteractionOutcome, Pose2D
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from geometry_msgs.msg import Pose2D as Pose2DMsg

ROBOT = 1
FOLLOWERS = (2, 3)
TICKS = 400
SPEED = 1.0


def _scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="external_robot_leader",
        simulation=SimulationParams(seed=3, dt=0.05, bt_tick_interval=5, max_ticks=TICKS),
        modules=ModuleConfig(global_planner="navmesh"),
        agents=[
            AgentConfig(agent_id=ROBOT, kind=1, policy="straight", agent_type="robot", spawn_pose=Pose2D(x=-10.0, y=0.0)),
            AgentConfig(agent_id=2, spawn_pose=Pose2D(x=-10.5, y=0.9), desired_velocity=1.3),
            AgentConfig(agent_id=3, spawn_pose=Pose2D(x=-10.5, y=-0.9), desired_velocity=1.3),
        ],
        interaction_scripts=[InteractionScript(tick=0, interaction_type="GROUP_WALK", participants=[ROBOT, *FOLLOWERS])],
        walls=[
            WallConfig(name="north", start=Pose2D(x=-13.0, y=5.0), end=Pose2D(x=13.0, y=5.0)),
            WallConfig(name="south", start=Pose2D(x=-13.0, y=-5.0), end=Pose2D(x=13.0, y=-5.0)),
        ],
    )


def _robot_pose(tick: int, dt: float) -> AgentStatesMsg:
    msg = AgentStatesMsg()
    stamp_ms = round(tick * dt * 1000)
    msg.header.stamp.sec = stamp_ms // 1000
    msg.header.stamp.nanosec = stamp_ms % 1000 * 1_000_000
    robot = AgentStateMsg()
    robot.agent_id = 424242
    robot.pose = Pose2DMsg(x=-10.0 + SPEED * tick * dt, y=0.0, theta=0.0)
    robot.radius = 0.3
    msg.agents.append(robot)
    return msg


def test_group_walks_abreast_of_a_robot_driven_through_world_state(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario(), node_name="test_external_robot_leader")

    abreast = 0
    for tick in range(TICKS):
        mgr._world_state_callback(_robot_pose(tick, mgr._dt))
        mgr.tick()
        groups = [i for i in mgr._interaction_manager.interactions.values() if i.type == int(InteractionType.GROUP_WALK)]
        assert len(groups) == 1 and groups[0].outcome == InteractionOutcome.ACTIVE
        assert set(groups[0].participants) == {ROBOT, *FOLLOWERS}, f"group broke at tick {tick}"

        rx = mgr._agents[ROBOT].state.pose.x
        followers = [mgr._agents[aid].state.pose for aid in FOLLOWERS]
        abreast += all(abs(p.x - rx) < 0.6 and abs(p.y) > 0.4 for p in followers)

    assert mgr._external_entities[424242].agent_id == ROBOT
    assert abreast > TICKS // 2
