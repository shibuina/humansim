from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.utils.scenario import load_scenario
from arena_humansim.utils.types import InteractionOutcome

_SCENARIO = Path(__file__).resolve().parents[2] / "config" / "scenarios" / "group_walk.yaml"
LEADER = 1
FOLLOWERS = (2, 3)


def _pose(mgr: AgentManager, aid: int) -> tuple[float, float]:
    p = mgr._agents[aid].state.pose
    return p.x, p.y


def test_group_walks_abreast_files_through_the_corridor_and_circles_up(manager_factory: Callable[..., AgentManager]) -> None:
    scenario = load_scenario(str(_SCENARIO))
    mgr = manager_factory(scenario, node_name="test_scripted_group_walk")

    abreast_in_hall = False
    for _ in range(scenario.simulation.max_ticks):
        mgr.tick()
        groups = [i for i in mgr._interaction_manager.interactions.values() if i.type == int(InteractionType.GROUP_WALK)]
        assert len(groups) == 1 and groups[0].outcome == InteractionOutcome.ACTIVE
        assert set(groups[0].participants) == {LEADER, *FOLLOWERS}

        lx, _ = _pose(mgr, LEADER)
        followers = [_pose(mgr, aid) for aid in FOLLOWERS]
        if -2.0 <= lx <= 2.0:
            assert all(abs(fy) < 0.7 and fx < lx for fx, fy in followers), f"not single file in the corridor: leader x={lx}, followers={followers}"
        if -10.0 <= lx <= -6.0 and all(abs(fx - lx) < 0.6 and abs(fy) > 0.4 for fx, fy in followers):
            abreast_in_hall = True

    assert abreast_in_hall
    lx, ly = _pose(mgr, LEADER)
    assert math.hypot(lx - 15.0, ly) < 0.5
    assert all(math.hypot(fx - lx, fy - ly) < 2.5 for fx, fy in (_pose(mgr, aid) for aid in FOLLOWERS))
