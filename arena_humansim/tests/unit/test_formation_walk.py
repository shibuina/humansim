from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.formation.anchor import AgentAnchor
from arena_humansim.core.formation.clearance import Clearance
from arena_humansim.core.formation.walk import WalkFormation
from arena_humansim.core.pool import AgentPool
from arena_humansim.perception.default import DefaultPerception
from arena_humansim.utils.types import Segments

DT = 0.05


def _group(
    agent_factory: Callable[..., BaseAgent],
    positions: dict[int, tuple[float, float]],
    walls: Segments | None = None,
    bystanders: dict[int, tuple[float, float]] | None = None,
    velocities: dict[int, tuple[float, float]] | None = None,
    **params: float,
) -> tuple[WalkFormation, dict[int, BaseAgent]]:
    agents = {aid: agent_factory(aid, x, y) for aid, (x, y) in {**positions, **(bystanders or {})}.items()}
    for aid, velocity in (velocities or {}).items():
        agents[aid].state.velocity = velocity
    anchor = AgentAnchor(pose_lookup=lambda aid: agents[aid].state.pose, agent_id=1)
    formation = WalkFormation(anchor=anchor, agent_lookup=agents.get, **params)
    if walls is not None:
        pool = AgentPool()
        for agent in agents.values():
            pool.add_agent(agent)
        DefaultPerception().compute_pool(pool)
        clearance = Clearance()
        clearance.attach(pool)
        clearance.set_walls(walls)
        formation.clearance = clearance
    for aid in positions:
        formation.on_join(aid)
    return formation, agents


def _walk_leader(formation: WalkFormation, leader: BaseAgent, steps: int, speed: float = 1.2) -> dict:
    leader.state.velocity = (speed, 0.0)
    targets: dict = {}
    for _ in range(steps):
        leader.state.pose.x += speed * DT
        targets = formation.tick(DT)
    return targets


CROWD = {10 + k: (x, y) for k, (x, y) in enumerate((x, side * 2.2) for x in (0.5, 1.5, 2.5) for side in (1.0, -1.0))} | {16: (4.0, 0.0)}


def _walk_in_place(formation: WalkFormation, leader: BaseAgent, seconds: float) -> None:
    leader.state.velocity = (1.2, 0.0)
    for _ in range(math.ceil(seconds / DT)):
        formation.tick(DT)


def test_open_space_followers_walk_abreast_on_their_own_side(agent_factory: Callable[..., BaseAgent]) -> None:
    formation, agents = _group(agent_factory, {1: (0.0, 0.0), 2: (-0.5, 1.0), 3: (-0.5, -1.0)})
    agents[1].state.velocity = (1.2, 0.0)

    targets = formation.tick(DT)

    left, right = formation.slot_of(2), formation.slot_of(3)
    assert left is not None and right is not None
    assert (left.x, left.y) == pytest.approx((0.0, formation.spacing))
    assert (right.x, right.y) == pytest.approx((0.0, -formation.spacing))
    assert (targets[2].x, targets[2].y) == pytest.approx((1.2 * formation.lead_time, formation.spacing))
    assert 1 not in targets


def test_trailing_follower_speeds_up_and_leader_waits(agent_factory: Callable[..., BaseAgent]) -> None:
    formation, agents = _group(agent_factory, {1: (0.0, 0.0), 2: (-6.0, 0.5)})
    agents[1].state.velocity = (1.2, 0.0)

    formation.tick(DT)
    speeds = formation.speeds()

    assert speeds[2] == pytest.approx(agents[2].params.max_velocity)
    assert speeds[1] == pytest.approx(agents[1].state.desired_velocity * formation.min_pace)


def test_narrow_corridor_files_followers_behind_the_leader(agent_factory: Callable[..., BaseAgent]) -> None:
    corridor = [((-20.0, 0.7), (20.0, 0.7)), ((-20.0, -0.7), (20.0, -0.7))]
    formation, agents = _group(agent_factory, {1: (0.0, 0.0), 2: (-0.9, 0.2), 3: (-1.8, -0.2)}, walls=corridor)

    _walk_leader(formation, agents[1], steps=40)

    lead_x = agents[1].state.pose.x
    slots = sorted((formation.slot_of(aid) for aid in (2, 3)), key=lambda p: -p.x)
    assert [s.y for s in slots] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert [lead_x - s.x for s in slots] == pytest.approx([formation.row_gap, 2 * formation.row_gap], abs=0.06)


def test_wide_corridor_keeps_followers_abreast(agent_factory: Callable[..., BaseAgent]) -> None:
    corridor = [((-20.0, 2.0), (20.0, 2.0)), ((-20.0, -2.0), (20.0, -2.0))]
    formation, agents = _group(agent_factory, {1: (0.0, 0.0), 2: (-0.5, 1.0), 3: (-0.5, -1.0)}, walls=corridor)

    _walk_leader(formation, agents[1], steps=20)

    lead_x = agents[1].state.pose.x
    assert [formation.slot_of(aid).x for aid in (2, 3)] == pytest.approx([lead_x, lead_x], abs=1e-6)


def test_dense_crowd_files_the_group_single_file(agent_factory: Callable[..., BaseAgent]) -> None:
    dense = {100 + k: (x, y) for k, (x, y) in enumerate((x, y) for x in (0.5, 1.5, 2.5, 3.5) for y in (-2.2, -1.4, 1.4, 2.2))}
    formation, agents = _group(agent_factory, {1: (0.0, 0.0), 2: (-0.5, 1.0), 3: (-0.5, -1.0)}, walls=[], bystanders=dense)

    _walk_in_place(formation, agents[1], seconds=5.0)

    lead_x = agents[1].state.pose.x
    slots = sorted((formation.slot_of(aid) for aid in (2, 3)), key=lambda p: -p.x)
    assert [s.y for s in slots] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert [lead_x - s.x for s in slots] == pytest.approx([formation.row_gap, 2 * formation.row_gap])


def test_stopped_leader_closes_the_group_into_an_inward_circle(agent_factory: Callable[..., BaseAgent]) -> None:
    formation, agents = _group(agent_factory, {1: (0.0, 0.0), 2: (-0.5, 1.0), 3: (-0.5, -1.0)})
    agents[1].state.velocity = (0.0, 0.0)

    steps = math.ceil(formation.stop_time / DT) + 1
    for _ in range(steps):
        targets = formation.tick(DT)

    lead = agents[1].state.pose
    ring = np.array([(lead.x, lead.y), *((targets[aid].x, targets[aid].y) for aid in (2, 3))])
    center = ring.mean(axis=0)
    radii = np.linalg.norm(ring - center, axis=1)
    assert radii == pytest.approx(np.full(3, radii[0]), abs=1e-6)
    for aid in (2, 3):
        facing = math.atan2(center[1] - targets[aid].y, center[0] - targets[aid].x)
        assert math.cos(targets[aid].theta - facing) == pytest.approx(1.0)
    assert formation.speeds() == {}


def test_stopped_circle_keeps_slots_off_a_nearby_wall(agent_factory: Callable[..., BaseAgent]) -> None:
    formation, agents = _group(agent_factory, {1: (0.0, 0.0), 2: (0.6, 0.3), 3: (0.6, -0.3)}, walls=[((1.5, -5.0), (1.5, 5.0))])
    agents[1].state.velocity = (0.0, 0.0)

    for _ in range(math.ceil(formation.stop_time / DT) + 1):
        targets = formation.tick(DT)

    assert all(targets[aid].x < 1.5 for aid in (2, 3))


def test_clearance_rays_stop_at_walls_and_crowd_counts_other_agents(agent_factory: Callable[..., BaseAgent]) -> None:
    pool = AgentPool()
    pool.add_agent(agent_factory(1, 0.0, 0.0))
    pool.add_agent(agent_factory(5, 3.0, 0.0))
    DefaultPerception().compute_pool(pool)
    clearance = Clearance()
    clearance.attach(pool)
    clearance.set_walls([((-1.0, 1.0), (1.0, 1.0))])

    free = clearance.free(np.zeros((3, 2)), np.array([(0.0, 1.0), (0.0, -1.0), (1.0, 0.0)]), 10.0)

    assert free == pytest.approx([1.0, 10.0, 10.0])
    assert clearance.crowds([1, 1, 1], np.zeros((3, 2)), np.array([4.0, 4.0, 2.0]), [(), (5,), ()]).tolist() == [1, 0, 0]


def test_moderate_crowd_trails_a_v_back_from_the_leader(agent_factory: Callable[..., BaseAgent]) -> None:
    formation, agents = _group(agent_factory, {1: (0.0, 0.0), 2: (-0.5, 1.0), 3: (-0.5, -1.0)}, walls=[], bystanders=CROWD)

    _walk_in_place(formation, agents[1], seconds=5.0)

    lead_x = agents[1].state.pose.x
    middle, wing = sorted((formation.slot_of(aid) for aid in (2, 3)), key=lambda p: abs(p.y))
    assert (middle.y, wing.y) == pytest.approx((formation.spacing, 2 * formation.spacing))
    assert lead_x - middle.x > 0.8 * formation.vee_depth
    assert wing.x == pytest.approx(lead_x)


def test_v_opens_away_from_oncoming_walkers(agent_factory: Callable[..., BaseAgent]) -> None:
    oncoming_left = {aid: (-1.0, 0.0) for aid, (_, y) in CROWD.items() if y > 0.0}
    formation, agents = _group(agent_factory, {1: (0.0, 0.0), 2: (-0.5, 1.0), 3: (-0.5, -1.0)}, walls=[], bystanders=CROWD, velocities=oncoming_left)

    _walk_in_place(formation, agents[1], seconds=5.0)

    assert all(formation.slot_of(aid).y < 0.0 for aid in (2, 3))


def test_empty_space_keeps_a_trio_abreast(agent_factory: Callable[..., BaseAgent]) -> None:
    formation, agents = _group(agent_factory, {1: (0.0, 0.0), 2: (-0.5, 1.0), 3: (-0.5, -1.0)}, walls=[])

    _walk_in_place(formation, agents[1], seconds=5.0)

    assert [formation.slot_of(aid).x for aid in (2, 3)] == pytest.approx([agents[1].state.pose.x] * 2)


def test_crowd_leaves_a_pair_side_by_side(agent_factory: Callable[..., BaseAgent]) -> None:
    formation, agents = _group(agent_factory, {1: (0.0, 0.0), 2: (-0.5, 1.0)}, walls=[], bystanders=CROWD)

    _walk_in_place(formation, agents[1], seconds=5.0)

    assert formation.slot_of(2).x == pytest.approx(agents[1].state.pose.x)


def test_circle_on_stop_off_leaves_followers_beside_the_stopped_leader(agent_factory: Callable[..., BaseAgent]) -> None:
    formation, agents = _group(agent_factory, {1: (0.0, 0.0), 2: (-0.5, 1.0), 3: (-0.5, -1.0)}, circle_on_stop=False)
    agents[1].state.velocity = (0.0, 0.0)

    for _ in range(math.ceil(formation.stop_time / DT) * 3):
        formation.tick(DT)

    slots = [formation.slot_of(aid) for aid in (2, 3)]
    assert [(s.x, s.y) for s in slots] == pytest.approx([(0.0, formation.spacing), (0.0, -formation.spacing)])
