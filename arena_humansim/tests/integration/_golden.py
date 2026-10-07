from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import attrs
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.logger import SimulationLogger
from arena_humansim.utils.scenario import (
    AgentTemplateModel,
    FlowScenarioConfig,
    ModuleConfig,
    RateKeyframeModel,
    ScenarioConfig,
    ShapeModel,
    SimulationParams,
    SinkAffinityModel,
    SinkScenarioConfig,
    SourceScenarioConfig,
    WallConfig,
)
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3
from rclpy.parameter import Parameter

from ._helpers import build_manager

GOLDEN_DIR = Path(__file__).parent / "golden"
SEED = 7
DT = 0.05
TICKS = 300
WALL_Y = 1.0
WALL_X = 8.0
FLOW_X = 11.0


@attrs.frozen
class SpawnSpec:
    x: float
    y: float
    theta: float
    goal_x: float
    goal_y: float
    agent_type: str = "adult"
    desired_velocity: float = 1.2


@attrs.frozen
class GoldenFixture:
    name: str
    scenario: Callable[[], ScenarioConfig]
    spawns: tuple[SpawnSpec, ...]
    ticks: int = TICKS

    @property
    def session_path(self) -> Path:
        return GOLDEN_DIR / self.name / "session.jsonl"


def _walls(half_length: float) -> list[WallConfig]:
    return [
        WallConfig(name="north", start=Pose2D(x=-half_length, y=WALL_Y), end=Pose2D(x=half_length, y=WALL_Y)),
        WallConfig(name="south", start=Pose2D(x=-half_length, y=-WALL_Y), end=Pose2D(x=half_length, y=-WALL_Y)),
    ]


def _corridor(name: str, local_planner: str, flow: FlowScenarioConfig | None = None, half_length: float = WALL_X) -> ScenarioConfig:
    return ScenarioConfig(
        name=name,
        simulation=SimulationParams(seed=SEED, dt=DT, max_ticks=TICKS),
        modules=ModuleConfig(local_planner=local_planner),
        walls=_walls(half_length),
        flow=flow,
    )


def _flow() -> FlowScenarioConfig:
    return FlowScenarioConfig(
        sources=[
            SourceScenarioConfig(
                pose=Pose2D(x=-FLOW_X, y=0.0, theta=0.0),
                shape=ShapeModel(type="circle", radius=0.3),
                type="poisson",
                rate_profile=[RateKeyframeModel(t=0.0, rate=0.6)],
                max_concurrent=4,
                agent_template=AgentTemplateModel(
                    desired_velocity_min=1.0,
                    desired_velocity_max=1.4,
                    agent_radius=0.25,
                    sink_affinity=[SinkAffinityModel(sink_idx=0, weight=1.0)],
                ),
            ),
        ],
        sinks=[
            SinkScenarioConfig(
                pose=Pose2D(x=FLOW_X, y=0.0, theta=0.0),
                shape=ShapeModel(type="circle", radius=0.5),
                absorption_radius=0.5,
            ),
        ],
    )


CORRIDOR_SPAWNS: tuple[SpawnSpec, ...] = (
    SpawnSpec(-5.0, -0.5, 0.0, 6.0, -0.5),
    SpawnSpec(-4.0, 0.2, 0.0, 6.0, 0.2),
    SpawnSpec(5.0, 0.5, math.pi, -6.0, 0.5),
    SpawnSpec(4.0, -0.2, math.pi, -6.0, -0.2),
    SpawnSpec(-2.0, 0.7, 0.0, 6.0, 0.7, "elder", 0.7),
    SpawnSpec(2.0, -0.7, math.pi, -6.0, -0.7, "elder", 0.7),
)

FIXTURES: tuple[GoldenFixture, ...] = (
    GoldenFixture("sfm_corridor", lambda: _corridor("sfm_corridor", "sfm"), CORRIDOR_SPAWNS),
    GoldenFixture("hsfm_corridor", lambda: _corridor("hsfm_corridor", "hsfm"), CORRIDOR_SPAWNS),
    GoldenFixture("flow_corridor", lambda: _corridor("flow_corridor", "sfm", _flow(), FLOW_X + 1.0), CORRIDOR_SPAWNS[2:4] + CORRIDOR_SPAWNS[5:6]),
)


def spawn(mgr: AgentManager, specs: tuple[SpawnSpec, ...]) -> list[int]:
    req = SpawnAgents.Request()
    for s in specs:
        msg = AgentStateMsg()
        msg.agent_id = 0
        msg.pose = RosPose2D(x=s.x, y=s.y, theta=s.theta)
        msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
        msg.desired_velocity = s.desired_velocity
        msg.radius = 0.0
        msg.agent_type = s.agent_type
        wp = WaypointMsg()
        wp.pose = RosPose2D(x=s.goal_x, y=s.goal_y, theta=0.0)
        msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
        req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return list(resp.spawned_ids)


def record_fixture(fixture: GoldenFixture, out_dir: Path) -> Path:
    log_dir = out_dir / fixture.name
    mgr = build_manager(
        fixture.scenario(),
        node_name=f"golden_record_{fixture.name}",
        extra_params=[Parameter("log_dir", Parameter.Type.STRING, str(log_dir))],
    )
    try:
        assert isinstance(mgr._sim_logger, SimulationLogger)
        spawn(mgr, fixture.spawns)
        for _ in range(fixture.ticks):
            mgr.tick()
        mgr._sim_logger.close()
    finally:
        mgr.destroy_node()
    return log_dir / "session.jsonl"


@attrs.frozen
class SessionStats:
    ticks: int
    spawned: int
    despawned: int
    max_agents: int
    max_displacement: float
    min_pair_distance: float
    min_wall_distance: float


def session_stats(path: Path) -> SessionStats:
    spawned = 0
    despawned = 0
    ticks = 0
    max_agents = 0
    first: dict[str, tuple[float, float]] = {}
    max_disp = 0.0
    min_pair = math.inf
    min_wall = math.inf
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            event = rec.get("event")
            if event == "spawn":
                spawned += 1
                continue
            if event == "despawn":
                despawned += 1
                continue
            ticks += 1
            pts = {aid: (a["pose"]["x"], a["pose"]["y"]) for aid, a in rec["agents"].items()}
            max_agents = max(max_agents, len(pts))
            for aid, (x, y) in pts.items():
                fx, fy = first.setdefault(aid, (x, y))
                max_disp = max(max_disp, math.hypot(x - fx, y - fy))
                min_wall = min(min_wall, WALL_Y - abs(y))
            ids = list(pts)
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    min_pair = min(min_pair, math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]))
    return SessionStats(ticks, spawned, despawned, max_agents, max_disp, min_pair, min_wall)


def record(out_dir: Path) -> None:
    for fixture in FIXTURES:
        path = record_fixture(fixture, out_dir)
        print(f"{fixture.name}: {path} {session_stats(path)}")


if __name__ == "__main__":
    import rclpy
    import rclpy.node
    from arena_humansim.utils.loggable import Loggable

    rclpy.init()
    host = rclpy.node.Node("golden_record_host")
    Loggable.init_logging(host)
    try:
        record(GOLDEN_DIR)
    finally:
        host.destroy_node()
        rclpy.shutdown()
