from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.utils.scenario import load_scenario
from arena_humansim.utils.types import InteractionOutcome

pytestmark = pytest.mark.slow

_SPARSE = Path(__file__).resolve().parents[2] / "config" / "evaluation" / "bt" / "sparse"


def _run(manager_factory: Callable[..., AgentManager], name: str, seconds: float, each_tick: Callable[[AgentManager], None]) -> None:
    mgr = manager_factory(load_scenario(str(_SPARSE / f"{name}.yaml")), node_name=f"test_eval_{name}")
    for _ in range(round(seconds / mgr._dt)):
        mgr.tick()
        each_tick(mgr)


def test_queue_use_passes_the_fountain_down_the_queue(manager_factory: Callable[..., AgentManager]) -> None:
    users: set[int] = set()

    def watch(mgr: AgentManager) -> None:
        for interaction in mgr._interaction_manager.interactions.values():
            if interaction.type == InteractionType.USE and interaction.outcome == InteractionOutcome.ACTIVE:
                users.update(interaction.participants)

    _run(manager_factory, "queue_use", 120.0, watch)
    assert len(users) >= 3


def test_static_service_serves_a_seeker(manager_factory: Callable[..., AgentManager]) -> None:
    served: set[int] = set()

    def watch(mgr: AgentManager) -> None:
        for interaction in mgr._interaction_manager.interactions.values():
            if interaction.type == InteractionType.SERVICE and interaction.outcome == InteractionOutcome.ACTIVE:
                served.update(pid for pid in interaction.participants if pid != interaction.provider)

    _run(manager_factory, "service_static", 120.0, watch)
    assert served


def test_mobile_escorts_carry_residents_across(manager_factory: Callable[..., AgentManager]) -> None:
    start: dict[int, float] = {}
    crossed: set[int] = set()

    def watch(mgr: AgentManager) -> None:
        for aid in (10, 11, 12, 20, 21, 22):
            x = mgr._agents[aid].state.pose.x
            start.setdefault(aid, x)
            if x * start[aid] < 0.0 and abs(x) > 3.0:
                crossed.add(aid)

    _run(manager_factory, "service_mobile", 120.0, watch)
    assert crossed == {10, 11, 12, 20, 21, 22}
