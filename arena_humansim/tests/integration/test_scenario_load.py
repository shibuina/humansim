from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import load_scenario

pytestmark = pytest.mark.slow

_SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "config" / "scenarios"
_SCENARIO_PATHS = sorted(_SCENARIOS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("scenario_path", _SCENARIO_PATHS, ids=lambda p: p.name)
def test_scenario_builds_and_ticks(manager_factory: Callable[..., AgentManager], scenario_path: Path) -> None:
    scenario = load_scenario(str(scenario_path))
    node_name = "test_scn_" + scenario_path.stem
    mgr = manager_factory(scenario, node_name=node_name)
    assert len(mgr._walls) == len({w.name for w in scenario.walls})
    for _ in range(10):
        mgr.tick()
    assert mgr._tick_count == 10
