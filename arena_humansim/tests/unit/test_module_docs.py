from __future__ import annotations

import re
from pathlib import Path

from arena_humansim.animation import MotionAnimation
from arena_humansim.collision import CollisionResolver
from arena_humansim.global_planner import GlobalPlanner
from arena_humansim.local_planner import LocalPlanner
from arena_humansim.occlusion import Occluder
from arena_humansim.perception import Perception

_PACKAGE = Path(__file__).parents[2]
_CROWD_PLANNERS = [name for name, info in LocalPlanner.info().items() if not info.robot_policy]


def _module_table() -> dict[str, list[str]]:
    rows = {}
    for line in (_PACKAGE.parent / "README.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 3 and cells[2].startswith("`"):
            layer = re.sub(r"\[(.*?)\]\(.*\)", r"\1", cells[0])
            rows[layer] = re.findall(r"`(\w+)`", cells[1])
    return rows


def test_readme_module_table_lists_what_is_registered() -> None:
    table = _module_table()
    assert table["Global Planner"] == GlobalPlanner.list_available()
    assert table["Local Planner"] == _CROWD_PLANNERS
    assert sorted(table["Perception"]) == sorted(Perception.list_available())
    assert sorted(table["Animation"]) == sorted(MotionAnimation.list_available())
    assert sorted(table["Collision"]) == sorted(CollisionResolver.list_available())
    assert sorted(table["Occlusion"]) == sorted(Occluder.list_available())


def test_local_planner_readme_has_a_row_per_crowd_planner() -> None:
    text = (_PACKAGE / "arena_humansim" / "local_planner" / "README.md").read_text()
    available = text.split("## Available")[1].split("\n## ")[0]
    rows = re.findall(r"^\| `(\w+)` \|", available, flags=re.M)
    assert sorted(rows) == sorted(_CROWD_PLANNERS)
