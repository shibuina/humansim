"""An effect can move an agent onto the waypoint driver while its behaviour tree still ticks; the
tree's Hold primitive must not write `movement.command` on a WaypointMovement (it has none - the
node died on it, 2026-08-30)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import py_trees
import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.agents.types import ParamDist
from arena_humansim.core.behavior.nodes import HoldNode
from arena_humansim.utils.types import BehaviorTreeMovement, Pose2D, WaypointMovement


def test_hold_node_keeps_time_but_writes_nothing_on_a_waypoint_agent(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=7, x=1.0, y=2.0)
    agent.movement = WaypointMovement(waypoints=[Pose2D(x=5.0, y=2.0)])
    node = HoldNode("hold", agent, duration_source=ParamDist(1.0), rng=np.random.default_rng(0), dt=0.5)
    node.setup()
    statuses = []
    for _ in range(4):
        node.tick_once()
        statuses.append(node.status)
    assert not hasattr(agent.movement, "command")
    assert isinstance(agent.movement, WaypointMovement) and agent.movement.waypoints
    assert statuses[0] == py_trees.common.Status.RUNNING
    assert py_trees.common.Status.SUCCESS in statuses, "the hold still runs out on time"


def test_hold_node_still_commands_a_tree_driven_agent(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=8, x=1.0, y=2.0)
    agent.movement = BehaviorTreeMovement()
    node = HoldNode("hold", agent, duration_source=ParamDist(1.0), rng=np.random.default_rng(0), dt=0.5)
    node.setup()
    node.tick_once()
    assert agent.movement.command is not None
    assert agent.movement.command.desired_velocity == 0.0
