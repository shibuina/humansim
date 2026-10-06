from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import load_scenario

pytestmark = pytest.mark.slow

_SPARSE = Path(__file__).resolve().parents[2] / "config" / "evaluation" / "bt" / "sparse"
_SIT = _SPARSE / "sit.yaml"


def test_autonomous_sit_action_seats_agents_and_restores_rest(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(load_scenario(str(_SIT)), node_name="test_autonomous_sit")
    im = mgr._interaction_manager
    start = {aid: agent.needs.needs["rest"].value for aid, agent in mgr._agents.items()}
    seated: set[int] = set()
    for _ in range(600):
        mgr.tick()
        seated.update(aid for aid in mgr._agents if im.posture_of(aid) == "seated")
    assert len(seated) >= 3
    assert sum(agent.needs.needs["rest"].value > start[aid] for aid, agent in mgr._agents.items()) >= 3


def test_object_counts_follow_the_live_interactions(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(load_scenario(str(_SPARSE / "compound.yaml")), node_name="test_object_counts")
    wk = mgr._world_knowledge
    used: set[str] = set()
    stale: list[tuple[int, str]] = []
    for tick in range(1800):
        mgr.tick()
        queued: dict[str, int] = {}
        members: dict[str, int] = {}
        for interaction in mgr._interaction_manager.interactions.values():
            if interaction.object_id:
                used.add(interaction.object_id)
                queued[interaction.object_id] = queued.get(interaction.object_id, 0) + interaction.contract.queue_length
                members[interaction.object_id] = members.get(interaction.object_id, 0) + len(interaction.participants)
        stale.extend((tick, oid) for oid in wk._objects if (wk.queue_length_for_object(oid), wk.participants_count_for_object(oid)) != (queued.get(oid, 0), members.get(oid, 0)))
    assert used
    assert not stale
