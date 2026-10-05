from __future__ import annotations

from typing import cast

import numpy as np
import pytest

pytest.importorskip("arena_humansim_msgs")
pytest.importorskip("rclpy")

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agent_manager import _CMD_LABEL_IDX, _CMD_LABELS, _flat, _group_by
from arena_humansim.utils.types import CommandType


class _Tagged:
    def __init__(self, tag: str, seq: int) -> None:
        self.tag = tag
        self.seq = seq


def _agents(*items: tuple[str, int]) -> list[BaseAgent]:
    return cast(list[BaseAgent], [_Tagged(t, s) for t, s in items])


def test_group_by_groups_by_key() -> None:
    items = _agents(("a", 1), ("b", 2), ("a", 3), ("c", 4), ("b", 5))
    groups = dict(_group_by(items, lambda x: cast(_Tagged, x).tag))
    seqs = {k: [cast(_Tagged, a).seq for a in v] for k, v in groups.items()}
    assert seqs == {"a": [1, 3], "b": [2, 5], "c": [4]}


def test_group_by_preserves_insertion_order_within_groups() -> None:
    items = _agents(("a", 1), ("a", 2), ("a", 3), ("a", 4))
    groups = dict(_group_by(items, lambda x: cast(_Tagged, x).tag))
    assert [cast(_Tagged, a).seq for a in groups["a"]] == [1, 2, 3, 4]


def test_group_by_empty_returns_empty() -> None:
    assert dict(_group_by(cast(list[BaseAgent], []), lambda _: 0)) == {}


def test_group_by_single_item_per_key() -> None:
    items = _agents(("a", 1), ("b", 2), ("c", 3))
    groups = dict(_group_by(items, lambda x: cast(_Tagged, x).tag))
    assert {k: len(v) for k, v in groups.items()} == {"a": 1, "b": 1, "c": 1}


@pytest.mark.parametrize("n", [0, 1, 7, 16, 17, 80])
def test_flat_returns_exactly_n_values(n: int) -> None:
    out = _flat("d", np.arange(n, dtype=np.float64))
    assert out.typecode == "d"
    assert list(out) == [float(i) for i in range(n)]


def test_flat_reads_strided_columns() -> None:
    pos = np.arange(12, dtype=np.float64).reshape(6, 2)
    assert list(_flat("d", pos[:, 0])) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert list(_flat("d", pos[[4, 1], 1])) == [9.0, 3.0]


@pytest.mark.parametrize(("code", "values"), [("B", [0, 1, 255]), ("H", [0, 7, 65535]), ("h", [-1, 0, 2]), ("i", [-5, 0, 1 << 30]), ("I", [0, 3, (1 << 32) - 1]), ("f", [0.5, -1.25, 100.0])])
def test_flat_casts_to_the_message_element_type(code: str, values: list[float]) -> None:
    out = _flat(code, np.array(values, dtype=np.float64 if code == "f" else np.int64))
    assert out.typecode == code
    assert list(out) == values


def test_cmd_label_table_covers_every_command_and_intr() -> None:
    assert _CMD_LABELS == (*(c.name for c in CommandType), "INTR")
    assert all(_CMD_LABELS[_CMD_LABEL_IDX[c]] == c.name for c in CommandType)
