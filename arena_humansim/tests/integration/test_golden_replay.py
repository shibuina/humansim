from __future__ import annotations

from collections.abc import Callable

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.replay import ReplayManager

from ._golden import FIXTURES, GoldenFixture, spawn


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f.name for f in FIXTURES])
def test_golden_session_replays_without_divergence(manager_factory: Callable[..., AgentManager], fixture: GoldenFixture) -> None:
    replay = ReplayManager()
    replay.load(str(fixture.session_path))
    assert replay.tick_count == fixture.ticks

    mgr = manager_factory(fixture.scenario(), node_name=f"golden_replay_{fixture.name}")
    spawn(mgr, fixture.spawns)

    result = replay.replay(mgr)
    assert result.success, f"replay diverged: {result.first_divergence}"
    assert result.total_ticks == fixture.ticks
