from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

from arena_humansim.utils.bag_io import extract_agent_states, policy_names

from tests.unit._bags import FLAT, LAYOUTS, NESTED, Agent, Frames, write_bag

_TRAJECTORY: Frames = [
    (0, [Agent(1, 0.0, 1.0, vx=0.1), Agent(-1, 3.0, 0.0, policy="", kind=1, radius=0.5)]),
    (50_000_000, [Agent(1, 0.5, 1.0, vx=0.1, name="alice", handedness="l"), Agent(2, 0.0, -1.0, vy=0.2, policy="orca"), Agent(-1, 3.1, 0.0, policy="", kind=1, radius=0.5)]),
    (100_000_000, [Agent(1, 1.0, 1.0, vx=0.1, policy="orca", name="alice", handedness="l"), Agent(-1, 3.2, 0.0, policy="", kind=1, radius=0.5)]),
]


def test_flat_and_nested_bags_extract_the_same_frame(tmp_path: Path) -> None:
    write_bag(tmp_path / NESTED, NESTED, _TRAJECTORY)
    write_bag(tmp_path / FLAT, FLAT, _TRAJECTORY)

    nested = extract_agent_states(tmp_path / NESTED)
    flat = extract_agent_states(tmp_path / FLAT)

    pd.testing.assert_frame_equal(nested, flat)
    assert list(flat.columns) == ["time", "agent_id", "x", "y", "vx", "vy", "radius", "planner"]
    assert flat["planner"].tolist() == ["sfm", "", "sfm", "orca", "", "orca", ""]
    assert flat["agent_id"].tolist() == [1, -1, 1, 2, -1, 1, -1]
    assert flat["time"].tolist() == [0.0, 0.0, 0.05, 0.05, 0.05, 0.1, 0.1]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_bag_with_only_empty_frames_extracts_empty(tmp_path: Path, layout: str) -> None:
    write_bag(tmp_path / "bag", layout, [(0, []), (50_000_000, [])])

    assert extract_agent_states(tmp_path / "bag").empty


@pytest.mark.parametrize("layout", LAYOUTS)
def test_empty_frames_around_agents_add_no_rows(tmp_path: Path, layout: str) -> None:
    write_bag(tmp_path / "bag", layout, [(0, []), (50_000_000, [Agent(4, 1.5, 2.5, vx=0.3)]), (100_000_000, [])])

    df = extract_agent_states(tmp_path / "bag")

    assert df.to_dict("records") == [{"time": 0.05, "agent_id": 4, "x": 1.5, "y": 2.5, "vx": 0.3, "vy": 0.0, "radius": 0.35, "planner": "sfm"}]


def test_flat_meta_written_after_its_frames_still_names_them(tmp_path: Path) -> None:
    write_bag(tmp_path / "bag", FLAT, _TRAJECTORY, late_meta=True)

    assert extract_agent_states(tmp_path / "bag")["planner"].tolist() == ["sfm", "", "sfm", "orca", "", "orca", ""]


def test_flat_frames_without_meta_have_empty_planner(tmp_path: Path) -> None:
    write_bag(tmp_path / "bag", FLAT, _TRAJECTORY, with_meta=False)

    df = extract_agent_states(tmp_path / "bag")

    assert df["planner"].tolist() == [""] * 7
    assert df["x"].tolist() == [0.0, 3.0, 0.5, 0.0, 3.1, 1.0, 3.2]


def test_policy_names_uses_the_latest_meta_at_or_before_each_frame() -> None:
    names = policy_names(
        meta_t=np.array([20, 10]),
        meta_policies=[["sfm", "orca"], ["sfm"]],
        frame_t=np.array([5, 10, 20]),
        counts=np.array([1, 2, 2]),
        policy_idx=np.array([0, 0, 1, 1, -1]),
    )

    assert names.tolist() == ["", "sfm", "", "orca", ""]
