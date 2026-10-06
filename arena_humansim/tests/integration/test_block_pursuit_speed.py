from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import load_scenario

_SCENARIO = """
name: block_pursuit_speed
simulation: {seed: 3, dt: 0.05, bt_tick_interval: 1, max_ticks: 100}
agent_types:
  chaser:
    extends: adult
    mode: behavior_tree
    desired_velocity: {mean: 1.0, std: 0.0}
    initial_sequence: default
    sequences:
      default:
        steps:
          chase:
            interaction: BLOCK
            target: 2
agents:
  - {agent_id: 1, agent_type: chaser, spawn_pose: {x: 0.0, y: 0.0, theta: 0.0}}
  - {agent_id: 2, spawn_pose: {x: 6.0, y: 0.0, theta: 0.0}, goal_sequence: [{x: 40.0, y: 0.0}], desired_velocity: 1.2}
"""


def test_block_pursuit_boost_reaches_the_local_planner(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    path = tmp_path / "block_pursuit_speed.yaml"
    path.write_text(_SCENARIO)
    mgr = manager_factory(load_scenario(str(path)), node_name="test_block_pursuit_speed")

    for _ in range(20):
        mgr.tick()

    chaser = mgr._agents[1]
    assert chaser.state.desired_velocity == pytest.approx(1.2 * 1.5)
    assert mgr._pool.desired_vel[mgr._pool.idx(1)] == pytest.approx(chaser.state.desired_velocity)
