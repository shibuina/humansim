from __future__ import annotations

HUMAN = 0
ROBOT = 1

_names: list[str] = ["human", "robot"]
_index: dict[str, int] = {"human": HUMAN, "robot": ROBOT}


def index(name: str) -> int:
    """Class index for a name, registering an unseen name in order of first sight; empty maps to HUMAN."""
    if not name:
        return HUMAN
    idx = _index.get(name)
    if idx is None:
        idx = len(_names)
        _names.append(name)
        _index[name] = idx
    return idx


def name(index: int) -> str:
    return _names[index]


def count() -> int:
    return len(_names)
