from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")
pytest.importorskip("pyarrow")

from arena_humansim.utils.evaluation.export import _process_trial

from tests.unit._bags import FLAT, NESTED, Agent, write_bag

_TRIAL = "nav_sparse_corridor__sfm__3"


def _trajectory() -> list[tuple[int, list[Agent]]]:
    frames = []
    for i in range(8):
        t = i * 0.1
        frames.append((i * 100_000_000, [Agent(1, t, 0.1 * t * t, vx=1.0, vy=0.2 * t), Agent(2, 4.0 - t, 1.0, vx=-1.0, policy="orca")]))
    return frames


def _export(root: Path, layout: str) -> tuple[pd.DataFrame, dict]:
    recordings = root / layout / "recordings"
    write_bag(recordings / _TRIAL / "bag", layout, _trajectory())
    states_dir = root / layout / "states"
    result = _process_trial(recordings, _TRIAL, states_dir, do_robots=False)
    assert result is not None
    (shard,) = states_dir.rglob("*.parquet")
    return pd.read_parquet(shard), result


def test_flat_and_nested_bags_export_the_same_shard(tmp_path: Path) -> None:
    nested_df, nested = _export(tmp_path, NESTED)
    flat_df, flat = _export(tmp_path, FLAT)

    pd.testing.assert_frame_equal(nested_df, flat_df)
    assert len(flat_df) == 16
    assert nested["n_rows"] == flat["n_rows"]
    assert nested["full_columns"] == flat["full_columns"]
    assert nested["full_dtypes"] == flat["full_dtypes"]
    for key, value in nested["kin_row"].items():
        other = flat["kin_row"][key]
        assert (isinstance(value, float) and math.isnan(value) and math.isnan(other)) or value == other
