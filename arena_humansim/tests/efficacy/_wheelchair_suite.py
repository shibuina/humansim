from __future__ import annotations

import itertools
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path

import attrs
import numba
import numpy as np
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.agents import sample_agent_type
from arena_humansim.core.locomotion import MODE_NORMAL
from arena_humansim.core.pool import KIND_HUMAN, AgentPool
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.scenario import AgentConfig, ScenarioConfig, load_scenario

from tests.integration._helpers import build_manager

SCENE_DIR = Path(__file__).resolve().parents[2] / "config" / "locomotion" / "wheelchair"
SCENES = ("doorway", "corridor_head_on_adult", "corridor_head_on_robot", "crowd_crossing", "corridor_u_turn", "robot_parked")
SUBJECT_TYPES = ("wheelchair_manual", "adult")
SUBJECT_ID = 1000
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
GOAL_RADIUS_M = 0.5
STUCK_SPEED = 0.05
VIA: dict[str, tuple[tuple[float, float], ...]] = {"doorway": ((0.0, 0.0),)}
METRICS = (
    "success_rate",
    "time_to_goal_s",
    "time_stuck_s",
    "path_efficiency",
    "min_clearance_m",
    "min_clearance_worst_m",
    "max_wall_penetration_m",
    "max_wall_penetration_worst_m",
    "recovery_events",
)
JSON_BEGIN = "=== wheelchair baseline json begin ==="
JSON_END = "=== wheelchair baseline json end ==="


@attrs.frozen
class RunMetrics:
    planner: str
    success: bool
    time_to_goal_s: float
    time_stuck_s: float
    path_efficiency: float
    min_clearance_m: float | None
    max_wall_penetration_m: float
    recovery_events: int


def scene_path(scene: str) -> Path:
    return SCENE_DIR / f"{scene}.yaml"


def subject_of(scenario: ScenarioConfig) -> AgentConfig:
    return next(a for a in scenario.agents if a.agent_id == SUBJECT_ID)


def budget_s(scene: str) -> float:
    """Seconds the subject has from its spawn tick to the scene's max_ticks."""
    scenario = load_scenario(str(scene_path(scene)))
    return (scenario.simulation.max_ticks - subject_of(scenario).spawn_tick) * scenario.simulation.dt


def configure(scenario: ScenarioConfig, subject_type: str, seed: int) -> ScenarioConfig:
    """Scenario under the seed with the subject's type swapped in and every scripted human's speed and radius drawn from its type."""
    rng = RNG(seed)
    agents = []
    for cfg in scenario.agents:
        agent = attrs.evolve(cfg, agent_type=subject_type) if cfg.agent_id == SUBJECT_ID else cfg
        if agent.kind == KIND_HUMAN:
            sampled = sample_agent_type(scenario.agent_types[agent.agent_type], rng.get_agent_substream(agent.agent_id, "params"), default_local_planner=scenario.modules.local_planner)
            agent = attrs.evolve(agent, desired_velocity=sampled.desired_velocity, agent_radius=sampled.agent_radius)
        agents.append(agent)
    return attrs.evolve(scenario, simulation=attrs.evolve(scenario.simulation, seed=seed), agents=agents)


def disk_centers(pool: AgentPool, rows: np.ndarray) -> np.ndarray:
    """Both capsule disk centers per row, shape (rows, 2, 2)."""
    axis = pool.axial_offset[rows, None] * np.column_stack([np.cos(pool.theta[rows]), np.sin(pool.theta[rows])])
    return np.stack([pool.pos[rows] + axis, pool.pos[rows] - axis], axis=1)


def wall_distance(points: np.ndarray, walls: np.ndarray) -> float:
    """Smallest distance from any point to any wall segment, walls as rows of (x0, y0, x1, y1)."""
    if walls.shape[0] == 0:
        return math.inf
    start = walls[:, :2]
    span = walls[:, 2:] - start
    rel = points[:, None, :] - start[None, :, :]
    t = np.clip(np.sum(rel * span[None, :, :], axis=2) / np.maximum(np.sum(span * span, axis=1), 1e-12), 0.0, 1.0)
    closest = start[None, :, :] + t[:, :, None] * span[None, :, :]
    return float(np.linalg.norm(points[:, None, :] - closest, axis=2).min())


def run(scene: str, subject_type: str, seed: int, each_tick: Callable[[AgentManager], None] | None = None) -> RunMetrics:
    """One seeded run of a scene, measured on the subject from its spawn until it reaches its final goal or the budget ends."""
    scenario = configure(load_scenario(str(scene_path(scene))), subject_type, seed)
    subject = subject_of(scenario)
    dt = scenario.simulation.dt
    budget_ticks = scenario.simulation.max_ticks - subject.spawn_tick
    walls = np.array([[w.start.x, w.start.y, w.end.x, w.end.y] for w in scenario.walls], dtype=np.float64).reshape(-1, 4)
    goals = [(g.x, g.y) for g in subject.goal_sequence]
    legs = [(subject.spawn_pose.x, subject.spawn_pose.y), *VIA.get(scene, ()), *goals]
    reference = sum(math.dist(a, b) for a, b in itertools.pairwise(legs)) - GOAL_RADIUS_M

    mgr = build_manager(scenario, node_name=f"wheelchair_suite_{scene}_{subject_type}_{seed}")
    try:
        for _ in range(subject.spawn_tick):
            mgr.tick()
        pool = mgr._pool
        prev = np.array(legs[0], dtype=np.float64)
        prev_mode = MODE_NORMAL
        travelled = 0.0
        stuck_ticks = 0
        clearance = math.inf
        penetration = 0.0
        recoveries = 0
        reached_ticks = 0
        planner = ""
        for tick in range(1, budget_ticks + 1):
            mgr.tick()
            i = pool.idx(SUBJECT_ID)
            planner = mgr._policy_names[int(pool.policy_idx[i])]
            pos = pool.pos[i].copy()
            step = float(np.hypot(*(pos - prev)))
            travelled += step
            prev = pos

            own = disk_centers(pool, np.array([i]))[0]
            others = np.flatnonzero(np.arange(pool.n) != i)
            if others.size:
                gaps = np.linalg.norm(own[None, :, None, :] - disk_centers(pool, others)[:, None, :, :], axis=3).min(axis=(1, 2))
                clearance = min(clearance, float((gaps - pool.agent_radius[i] - pool.agent_radius[others]).min()))
            penetration = max(penetration, float(pool.agent_radius[i]) - wall_distance(own, walls))

            mode = int(mgr._locomotion.mode[i])
            recoveries += int(prev_mode == MODE_NORMAL and mode != MODE_NORMAL)
            prev_mode = mode

            if each_tick is not None:
                each_tick(mgr)
            index = mgr._agents[SUBJECT_ID].movement.index
            goal_dist = math.dist(pos, goals[index])
            if index == len(goals) - 1 and goal_dist < GOAL_RADIUS_M:
                reached_ticks = tick
                break
            if goal_dist > GOAL_RADIUS_M and step / dt < STUCK_SPEED:
                stuck_ticks += 1
    finally:
        mgr.destroy_node()

    success = reached_ticks > 0
    return RunMetrics(
        planner=planner,
        success=success,
        time_to_goal_s=(reached_ticks if success else budget_ticks) * dt,
        time_stuck_s=stuck_ticks * dt,
        path_efficiency=min(1.0, reference / travelled) if success else 0.0,
        min_clearance_m=clearance if math.isfinite(clearance) else None,
        max_wall_penetration_m=penetration,
        recovery_events=recoveries,
    )


def aggregate(runs: list[RunMetrics]) -> dict[str, float | None]:
    clearances = [r.min_clearance_m for r in runs if r.min_clearance_m is not None]
    return {
        "success_rate": sum(r.success for r in runs) / len(runs),
        "time_to_goal_s": statistics.median(r.time_to_goal_s for r in runs),
        "time_stuck_s": statistics.median(r.time_stuck_s for r in runs),
        "path_efficiency": statistics.median(r.path_efficiency for r in runs),
        "min_clearance_m": statistics.median(clearances) if clearances else None,
        "min_clearance_worst_m": min(clearances) if clearances else None,
        "max_wall_penetration_m": statistics.median(r.max_wall_penetration_m for r in runs),
        "max_wall_penetration_worst_m": max(r.max_wall_penetration_m for r in runs),
        "recovery_events": statistics.median(r.recovery_events for r in runs),
    }


@cache
def scene_runs(scene: str) -> dict[str, list[RunMetrics]]:
    """Runs per subject type over the fixed seed set, computed once per process."""
    return {subject_type: [run(scene, subject_type, seed) for seed in SEEDS] for subject_type in SUBJECT_TYPES}


def scene_aggregates(scene: str) -> dict[str, dict[str, float | None]]:
    return {subject_type: aggregate(runs) for subject_type, runs in scene_runs(scene).items()}


def machine_class() -> str:
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "")
    return f"{platform.machine()} / {cpu} / numpy {np.__version__} / numba {numba.__version__}"


def baseline() -> dict[str, object]:
    return {
        "machine": machine_class(),
        "seeds": list(SEEDS),
        "dt": load_scenario(str(scene_path(SCENES[0]))).simulation.dt,
        "budgets_s": {scene: budget_s(scene) for scene in SCENES},
        "scenes": {scene: scene_aggregates(scene) for scene in SCENES},
    }


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{round(value, 3) + 0.0:.3f}"


def markdown_tables(scenes: dict[str, dict[str, dict[str, float | None]]]) -> str:
    """One markdown table per metric, scenes as rows and subject types as columns."""
    blocks = []
    for metric in METRICS:
        rows = [f"### {metric}", "", "| scene | " + " | ".join(SUBJECT_TYPES) + " |", "|---|" + "---|" * len(SUBJECT_TYPES)]
        rows.extend(f"| {scene} | " + " | ".join(_cell(per_type[t][metric]) for t in SUBJECT_TYPES) + " |" for scene, per_type in scenes.items())
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def main() -> None:
    import rclpy
    from arena_humansim.utils.loggable import Loggable

    rclpy.init()
    node = rclpy.create_node("wheelchair_suite")
    Loggable.init_logging(node)
    try:
        start = time.perf_counter()
        result = baseline()
        wall = time.perf_counter() - start
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(markdown_tables(result["scenes"]))
    print()
    print(f"wall time: {wall:.1f} s for {len(SCENES) * len(SUBJECT_TYPES) * len(SEEDS)} runs")
    print(JSON_BEGIN)
    print(json.dumps(result, indent=2))
    print(JSON_END)


if __name__ == "__main__":
    main()
