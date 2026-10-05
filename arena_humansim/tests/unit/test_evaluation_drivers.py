from __future__ import annotations

from pathlib import Path

import pandas as pd

from arena_humansim.local_planner import LocalPlanner, robot
from arena_humansim.utils.evaluation.buckets import DRIVER_CLASS, DRIVER_CLASS_FINE
from arena_humansim.utils.evaluation.cli.benchmark import _discover_ped_planners, _discover_robot_policies
from arena_humansim.utils.evaluation.plots import DRIVER_COLOR, DRIVER_LABEL, DRIVER_ORDER, POLICY_LABEL, POLICY_ORDER, _family_edges, pair_matrix


def test_every_registered_planner_declares_its_info() -> None:
    assert list(LocalPlanner.info()) == LocalPlanner.list_available()


def test_evaluation_tables_follow_the_registry() -> None:
    assert sorted(DRIVER_ORDER) == _discover_ped_planners()
    assert sorted(POLICY_ORDER) == _discover_robot_policies()
    assert sorted(POLICY_ORDER) == sorted(p.name for p in Path(robot.__file__).parent.iterdir() if p.is_dir() and not p.name.startswith("_"))
    for table in (DRIVER_CLASS, DRIVER_CLASS_FINE, DRIVER_LABEL, DRIVER_COLOR):
        assert set(table) == set(DRIVER_ORDER)
    assert set(POLICY_LABEL) == set(POLICY_ORDER)


def test_driver_classes_binarize_the_families() -> None:
    assert DRIVER_CLASS_FINE["helbing"] == "force"
    assert DRIVER_CLASS_FINE["pedvo"] == "geometric"
    assert DRIVER_CLASS["pedvo"] == "classical"
    assert DRIVER_CLASS["socialgail"] == "learned"


def test_pair_matrix_keeps_only_recorded_drivers_in_registry_order() -> None:
    pairwise = pd.DataFrame(
        {
            "p1": ["orca", "sfm", "sfm"],
            "p2": ["straight", "orca", "straight"],
            "scenario": ["a", "a", "a"],
            "bucket": ["nav", "nav", "nav"],
            "hausdorff": [1.0, 2.0, 3.0],
        }
    )
    mat = pair_matrix(pairwise, "nav")
    assert list(mat.index) == ["sfm", "orca", "straight"]
    assert mat.loc["orca", "sfm"] == 2.0
    assert mat.loc["straight", "sfm"] == 3.0
    assert _family_edges(list(mat.index)) == [1, 2]
