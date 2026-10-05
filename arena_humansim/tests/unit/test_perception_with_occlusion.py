from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import SampledParams, SampledPerception
from arena_humansim.core.pool import AgentPool
from arena_humansim.occlusion import Occluder
from arena_humansim.occlusion.bitmap import BitmapOccluder
from arena_humansim.perception.default import DefaultPerception
from arena_humansim.utils.benchmark import generate_maze
from arena_humansim.utils.types import AgentState, Pose2D, Segments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _params(
    vision_range: float = 10.0,
    vision_fov: float = 360.0,
    proximity_sense: float = 0.0,
    vision_occlusion: bool = True,
) -> SampledParams:
    return SampledParams(
        name="adult",
        desired_velocity=1.1,
        agent_radius=0.25,
        max_velocity=1.5,
        max_acceleration=1.5,
        max_deceleration=2.5,
        min_turning_radius=0.3,
        pivot_angular_velocity=2.0,
        reaction_time=0.4,
        personal_space_min=0.6,
        perception=SampledPerception(
            vision_range=vision_range,
            vision_fov=vision_fov,
            proximity_sense=proximity_sense,
            vision_occlusion=vision_occlusion,
        ),
        local_planner_params={
            "relaxation_time": 0.5,
            "repulsion_strength": 2.1,
            "repulsion_range": 0.3,
            "anisotropy": 0.5,
        },
    )


def _make_agent(
    agent_id: int,
    x: float,
    y: float,
    theta: float = 0.0,
    vision_range: float = 10.0,
    vision_fov: float = 360.0,
    proximity_sense: float = 0.0,
    vision_occlusion: bool = True,
) -> BaseAgent:
    state = AgentState(
        agent_id=agent_id,
        pose=Pose2D(x=x, y=y, theta=theta),
        velocity=(0.0, 0.0),
        desired_velocity=1.3,
    )
    return BaseAgent(
        state=state,
        params=_params(
            vision_range=vision_range,
            vision_fov=vision_fov,
            proximity_sense=proximity_sense,
            vision_occlusion=vision_occlusion,
        ),
        global_planner=cast(Any, None),
        local_planner=cast(Any, None),
        animation=cast(Any, None),
    )


def _pool_from(agents: list[BaseAgent]) -> AgentPool:
    pool = AgentPool(capacity=max(8, len(agents)))
    for ag in agents:
        pool.add_agent(ag)
    return pool


def _neighbors(pool: AgentPool, row: int) -> set[int]:
    indptr = pool.neighbor_indptr
    indices = pool.neighbor_indices
    return set(int(x) for x in indices[indptr[row] : indptr[row + 1]])


def _wall_at_x(x: float, half_len: float = 5.0) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [((x, -half_len), (x, half_len))]


# ---------------------------------------------------------------------------
# Regression baseline: no occluder produces same CSR as before this feature
# ---------------------------------------------------------------------------


def test_no_occluder_matches_baseline_dense() -> None:
    # Snapshot: two close agents, no occluder, omnidirectional. Both must see each other.
    agents = [
        _make_agent(1, 0.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(2, 2.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(3, 20.0, 0.0, vision_range=10.0, vision_fov=360.0),
    ]
    pool = _pool_from(agents)
    DefaultPerception().compute_pool(pool)

    assert 1 in _neighbors(pool, 0)
    assert 0 in _neighbors(pool, 1)
    # agent 3 is 20 m away - outside range for agents 1 and 2
    assert 2 not in _neighbors(pool, 0)
    assert 2 not in _neighbors(pool, 1)


def test_no_occluder_matches_baseline_kdtree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DefaultPerception, "_SMALL_N_THRESHOLD", 4)
    agents = [
        _make_agent(1, 0.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(2, 2.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(3, 4.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(4, 6.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(5, 50.0, 0.0, vision_range=10.0, vision_fov=360.0),
    ]
    pool = _pool_from(agents)
    DefaultPerception().compute_pool(pool)

    assert 1 in _neighbors(pool, 0)
    assert 0 in _neighbors(pool, 1)
    assert 4 not in _neighbors(pool, 0)
    assert 4 not in _neighbors(pool, 1)


# ---------------------------------------------------------------------------
# Bitmap occluder: agents either side of a wall are not visible to each other
# ---------------------------------------------------------------------------


def test_wall_blocks_agents_in_range_and_fov_dense() -> None:
    # Agent 1 at x=-2, agent 2 at x=2; wall at x=0. Both omni, long range.
    # Exercises the dense path (N=2 <= 64).
    occ = BitmapOccluder()
    occ.set_walls(_wall_at_x(0.0))
    agents = [
        _make_agent(1, -2.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(2, 2.0, 0.0, vision_range=10.0, vision_fov=360.0),
    ]
    pool = _pool_from(agents)
    DefaultPerception(occluder=occ).compute_pool(pool)

    assert 1 not in _neighbors(pool, 0)
    assert 0 not in _neighbors(pool, 1)


def test_wall_does_not_block_agents_on_same_side_dense() -> None:
    occ = BitmapOccluder()
    occ.set_walls(_wall_at_x(10.0))
    agents = [
        _make_agent(1, 0.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(2, 3.0, 0.0, vision_range=10.0, vision_fov=360.0),
    ]
    pool = _pool_from(agents)
    DefaultPerception(occluder=occ).compute_pool(pool)

    assert 1 in _neighbors(pool, 0)
    assert 0 in _neighbors(pool, 1)


# ---------------------------------------------------------------------------
# Proximity does NOT bypass LOS with occlusion ON
# ---------------------------------------------------------------------------


def test_prox_within_wall_not_visible_dense() -> None:
    # 0.5 m apart, well within proximity_sense=1.5, but wall between them.
    occ = BitmapOccluder()
    occ.set_walls(_wall_at_x(0.0))
    agents = [
        _make_agent(1, -0.3, 0.0, vision_range=10.0, vision_fov=360.0, proximity_sense=1.5),
        _make_agent(2, 0.3, 0.0, vision_range=10.0, vision_fov=360.0, proximity_sense=1.5),
    ]
    pool = _pool_from(agents)
    DefaultPerception(occluder=occ).compute_pool(pool)

    assert 1 not in _neighbors(pool, 0)
    assert 0 not in _neighbors(pool, 1)


def test_prox_no_wall_visible_dense() -> None:
    # Same distance, no wall: proximity must still grant visibility (f-formation case).
    agents = [
        _make_agent(1, -0.3, 0.0, theta=0.0, vision_range=5.0, vision_fov=60.0, proximity_sense=1.5),
        _make_agent(2, 0.3, 0.0, theta=0.0, vision_range=5.0, vision_fov=60.0, proximity_sense=1.5),
    ]
    pool = _pool_from(agents)
    DefaultPerception().compute_pool(pool)

    assert 1 in _neighbors(pool, 0)
    assert 0 in _neighbors(pool, 1)


# ---------------------------------------------------------------------------
# set_walls([]) with bitmap occluder -> identical to no occluder
# ---------------------------------------------------------------------------


def test_bitmap_empty_walls_matches_no_occluder() -> None:
    agents = [
        _make_agent(1, 0.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(2, 3.0, 0.0, vision_range=10.0, vision_fov=360.0),
    ]

    pool_none = _pool_from(agents)
    DefaultPerception().compute_pool(pool_none)

    occ = BitmapOccluder()
    occ.set_walls([])
    pool_occ = _pool_from(agents)
    DefaultPerception(occluder=occ).compute_pool(pool_occ)

    np.testing.assert_array_equal(pool_none.neighbor_indptr, pool_occ.neighbor_indptr)
    np.testing.assert_array_equal(pool_none.neighbor_indices, pool_occ.neighbor_indices)


# ---------------------------------------------------------------------------
# Per-agent vision_occlusion override
# ---------------------------------------------------------------------------


def test_vision_occlusion_false_agent_sees_through_wall_dense() -> None:
    # Agent 1 has vision_occlusion=False -> ignores LOS check, sees agent 2 through wall.
    # Agent 2 has vision_occlusion=True -> blocked by the wall, cannot see agent 1.
    # Exercises the dense path (N=2).
    occ = BitmapOccluder()
    occ.set_walls(_wall_at_x(0.0))
    agents = [
        _make_agent(1, -2.0, 0.0, vision_range=10.0, vision_fov=360.0, vision_occlusion=False),
        _make_agent(2, 2.0, 0.0, vision_range=10.0, vision_fov=360.0, vision_occlusion=True),
    ]
    pool = _pool_from(agents)
    DefaultPerception(occluder=occ).compute_pool(pool)

    # Agent 1 (idx 0) sees through walls - agent 2 must be in its neighbors.
    assert 1 in _neighbors(pool, 0)
    # Agent 2 (idx 1) respects LOS - agent 1 is on the other side of the wall.
    assert 0 not in _neighbors(pool, 1)


def test_vision_occlusion_false_agent_sees_through_wall_kdtree(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same scene, forced kdtree path (N > 64 threshold set to 4).
    monkeypatch.setattr(DefaultPerception, "_SMALL_N_THRESHOLD", 4)
    occ = BitmapOccluder()
    occ.set_walls(_wall_at_x(0.0))
    agents = [
        _make_agent(1, -2.0, 0.0, vision_range=10.0, vision_fov=360.0, vision_occlusion=False),
        _make_agent(2, 2.0, 0.0, vision_range=10.0, vision_fov=360.0, vision_occlusion=True),
        _make_agent(3, -4.0, 0.0, vision_range=10.0, vision_fov=360.0, vision_occlusion=True),
        _make_agent(4, 4.0, 0.0, vision_range=10.0, vision_fov=360.0, vision_occlusion=True),
        _make_agent(5, -6.0, 0.0, vision_range=10.0, vision_fov=360.0, vision_occlusion=True),
    ]
    pool = _pool_from(agents)
    DefaultPerception(occluder=occ).compute_pool(pool)

    # Agent 1 (idx 0, vision_occlusion=False) must see agents on the other side.
    assert 1 in _neighbors(pool, 0)
    # Agent 2 (idx 1, vision_occlusion=True) cannot see agent 1 through the wall.
    assert 0 not in _neighbors(pool, 1)


# ---------------------------------------------------------------------------
# Dense vs kdtree agreement under occlusion
# ---------------------------------------------------------------------------


def test_dense_kdtree_agreement_with_occlusion(monkeypatch: pytest.MonkeyPatch) -> None:
    # 5 agents: wall at x=0 separates left (negative x) from right (positive x).
    # Force dense path first, then kdtree path; neighbor sets must agree.
    wall = _wall_at_x(0.0)

    agents = [
        _make_agent(1, -3.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(2, -1.5, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(3, 1.5, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(4, 3.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(5, -5.0, 1.0, vision_range=10.0, vision_fov=360.0),
    ]

    occ_dense = BitmapOccluder()
    occ_dense.set_walls(wall)
    pool_dense = _pool_from(agents)
    # N=5 <= default threshold (64): dense path
    DefaultPerception(occluder=occ_dense).compute_pool(pool_dense)

    occ_kdtree = BitmapOccluder()
    occ_kdtree.set_walls(wall)
    pool_kdtree = _pool_from(agents)
    monkeypatch.setattr(DefaultPerception, "_SMALL_N_THRESHOLD", 4)
    # N=5 > patched threshold (4): kdtree path
    DefaultPerception(occluder=occ_kdtree).compute_pool(pool_kdtree)

    for row in range(len(agents)):
        dense_nb = _neighbors(pool_dense, row)
        kdtree_nb = _neighbors(pool_kdtree, row)
        assert dense_nb == kdtree_nb, f"row {row}: dense={dense_nb} kdtree={kdtree_nb}"


# ---------------------------------------------------------------------------
# kdtree path with occlusion: wall blocks and same-side visibility
# ---------------------------------------------------------------------------


def test_wall_blocks_across_kdtree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DefaultPerception, "_SMALL_N_THRESHOLD", 4)
    occ = BitmapOccluder()
    occ.set_walls(_wall_at_x(0.0))
    agents = [
        _make_agent(1, -2.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(2, 2.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(3, -4.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(4, 4.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(5, -6.0, 0.0, vision_range=10.0, vision_fov=360.0),
    ]
    pool = _pool_from(agents)
    DefaultPerception(occluder=occ).compute_pool(pool)

    # Agents on opposite sides of the wall must not be neighbors.
    assert 1 not in _neighbors(pool, 0)  # agent 1 (left) cannot see agent 2 (right)
    assert 0 not in _neighbors(pool, 1)  # agent 2 (right) cannot see agent 1 (left)

    # Agents on the same side must still see each other.
    assert 2 in _neighbors(pool, 0)  # agents 1 and 3 both left
    assert 0 in _neighbors(pool, 2)


def test_same_side_wall_visible_kdtree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DefaultPerception, "_SMALL_N_THRESHOLD", 4)
    occ = BitmapOccluder()
    occ.set_walls(_wall_at_x(10.0))  # wall far to the right, not between the agents
    agents = [
        _make_agent(1, 0.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(2, 3.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(3, 6.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(4, -3.0, 0.0, vision_range=10.0, vision_fov=360.0),
        _make_agent(5, -6.0, 0.0, vision_range=10.0, vision_fov=360.0),
    ]
    pool = _pool_from(agents)
    DefaultPerception(occluder=occ).compute_pool(pool)

    assert 1 in _neighbors(pool, 0)
    assert 0 in _neighbors(pool, 1)


class _RecordingOccluder(Occluder):
    def __init__(self, inner: BitmapOccluder) -> None:
        self._inner = inner
        self.calls: list[tuple[tuple[float, float], tuple[float, float], bool]] = []

    def set_walls(self, segments: Segments) -> None:
        self._inner.set_walls(segments)

    def clear(self, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
        result = self._inner.clear(p_a, p_b)
        for a, b, ok in zip(p_a, p_b, result, strict=True):
            self.calls.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1])), bool(ok)))
        return result


def _maze_segments() -> Segments:
    return [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in generate_maze(10, seed=42).walls]


@pytest.mark.parametrize("n", [40, 300], ids=["dense", "kdtree"])
@pytest.mark.parametrize("seed", [0, 1])
def test_line_of_sight_cast_once_per_unordered_pair_and_gates_occluding_observers(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 20.0, size=(n, 2))
    theta = rng.uniform(-np.pi, np.pi, size=n)
    vision_fov = rng.choice([90.0, 180.0, 210.0, 360.0], size=n)
    vision_range = rng.uniform(2.0, 10.0, size=n)
    proximity_sense = rng.choice([0.0, 0.5, 1.0], size=n)
    vision_occlusion = rng.random(n) < 0.7

    agents = [
        _make_agent(
            i + 1,
            float(pos[i, 0]),
            float(pos[i, 1]),
            theta=float(theta[i]),
            vision_range=float(vision_range[i]),
            vision_fov=float(vision_fov[i]),
            proximity_sense=float(proximity_sense[i]),
            vision_occlusion=bool(vision_occlusion[i]),
        )
        for i in range(n)
    ]

    pool_free = _pool_from(agents)
    DefaultPerception().compute_pool(pool_free)

    inner = BitmapOccluder()
    recorder = _RecordingOccluder(inner)
    recorder.set_walls(_maze_segments())
    pool = _pool_from(agents)
    DefaultPerception(occluder=recorder).compute_pool(pool)

    index_of = {(float(pos[i, 0]), float(pos[i, 1])): i for i in range(n)}
    clear_by_pair: dict[frozenset[int], bool] = {}
    for p_a, p_b, ok in recorder.calls:
        pair = frozenset((index_of[p_a], index_of[p_b]))
        assert len(pair) == 2
        assert pair not in clear_by_pair, f"pair {sorted(pair)} cast twice"
        clear_by_pair[pair] = ok

    assert any(clear_by_pair.values())
    assert not all(clear_by_pair.values())

    for r in range(n):
        candidates = _neighbors(pool_free, r)
        kept = _neighbors(pool, r)
        assert kept <= candidates
        if not vision_occlusion[r]:
            assert kept == candidates, f"row {r} ignores occlusion but lost {sorted(candidates - kept)}"
            continue
        for c in candidates:
            pair = frozenset((r, c))
            assert pair in clear_by_pair, f"candidate pair {(r, c)} never cast"
            assert (c in kept) == clear_by_pair[pair], f"row {r} neighbor {c}: kept={c in kept} clear={clear_by_pair[pair]}"
