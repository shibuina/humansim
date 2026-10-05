from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

from arena_humansim.utils.evaluation import bag_cache

from tests.unit._bags import LAYOUTS, Agent, write_bag


def _record(trial: Path, layout: str, x: float, mtime: float) -> None:
    bag = trial / "bag"
    shutil.rmtree(bag, ignore_errors=True)
    trial.mkdir(parents=True, exist_ok=True)
    write_bag(bag, layout, [(0, [Agent(1, x, 0.0)])])
    os.utime(bag / "metadata.yaml", (mtime, mtime))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_rerecorded_trial_invalidates_multi_cache(tmp_path: Path, layout: str) -> None:
    sweep = tmp_path / "sweep"
    trial = sweep / "scen__sfm__1"
    _record(trial, layout, x=1.0, mtime=1_000.0)

    first = bag_cache.load_multi([sweep])
    assert first["sweep"]["scen__sfm__1"]["x"].iloc[0] == 1.0
    assert first["sweep"]["scen__sfm__1"]["planner"].iloc[0] == "sfm"

    _record(trial, layout, x=2.0, mtime=(sweep / "_sweep_extracted.pkl").stat().st_mtime + 100.0)

    second = bag_cache.load_multi([sweep])
    assert second["sweep"]["scen__sfm__1"]["x"].iloc[0] == 2.0
