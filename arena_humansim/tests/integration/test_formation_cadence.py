from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.formation import Formation
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.scenario import load_scenario
from arena_humansim.utils.types import BehaviorTreeMovement, InteractionOutcome, Pose2D, SeekSpec

_GROUP_WALK = Path(__file__).resolve().parents[2] / "config" / "scenarios" / "group_walk.yaml"


@dataclass
class _FakeParams:
    reaction_time: float = 0.4
    personal_space_min: float = 0.6
    max_velocity: float = 1.8


@dataclass
class _FakeState:
    agent_id: int = 0
    pose: Pose2D = field(default_factory=Pose2D)
    velocity: tuple[float, float] = (0.0, 0.0)
    desired_velocity: float = 1.2
    kind: int = 0


@dataclass
class _FakeAgent:
    state: _FakeState
    params: _FakeParams = field(default_factory=_FakeParams)
    movement: BehaviorTreeMovement = field(default_factory=BehaviorTreeMovement)


class _Recorder(Formation):
    def __init__(self) -> None:
        self.ticks: list[float] = []

    def on_join(self, agent_id: int, *, participant: bool = True) -> None:
        del agent_id, participant

    def on_leave(self, agent_id: int) -> None:
        del agent_id

    def tick(self, dt: float) -> dict[int, Pose2D]:
        self.ticks.append(dt)
        return {}


def _walking_pair() -> tuple[InteractionManager, dict[int, _FakeAgent], int]:
    agents = {aid: _FakeAgent(state=_FakeState(agent_id=aid, pose=Pose2D(x=float(aid), y=0.0))) for aid in (1, 2)}
    mgr = InteractionManager(RNG(0))
    mgr.set_context(world_knowledge=WorldKnowledge(), agent_lookup=lambda aid: agents.get(aid))
    interaction = mgr._create_interaction(creator_id=1, spec=SeekSpec(interaction_type=InteractionType.GROUP_WALK))
    assert mgr.accept(2, interaction.id) is True
    assert interaction.outcome == InteractionOutcome.ACTIVE
    return mgr, agents, interaction.id


def test_formations_advance_on_formation_ticks_by_the_time_since_the_last() -> None:
    mgr, _, iid = _walking_pair()
    recorder = _Recorder()
    mgr.interactions[iid].contract.formation = recorder
    for _ in range(4):
        mgr.update({}, dt=0.05, formations=False)
    assert recorder.ticks == []
    mgr.update({}, dt=0.05, formations=True)
    mgr.update({}, dt=0.05, formations=True)
    assert recorder.ticks == pytest.approx([0.25, 0.05])


def test_drift_eviction_waits_for_a_formation_tick() -> None:
    mgr, agents, iid = _walking_pair()
    mgr.update({}, dt=0.05, formations=True)
    agents[2].state.pose = Pose2D(x=20.0, y=0.0)
    for _ in range(4):
        mgr.update({}, dt=0.05, formations=False)
    assert mgr.interactions[iid].participants == [1, 2]
    mgr.update({}, dt=0.05, formations=True)
    assert iid not in mgr.interactions
    assert agents[2].movement.last_outcome == InteractionOutcome.INTERRUPTED


def test_agent_manager_ticks_formations_right_before_each_decision_tick(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(load_scenario(str(_GROUP_WALK)), node_name="test_formation_cadence")
    im = mgr._interaction_manager
    refreshed: list[int] = []
    for _ in range(20):
        before = im._formation_targets
        mgr.tick()
        if im._formation_targets is not before:
            refreshed.append(mgr._tick_count)
    interval = mgr._bt_tick_interval
    assert refreshed == [t for t in range(1, 21) if t % interval == 0]
