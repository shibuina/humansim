from __future__ import annotations

import pytest

from arena_humansim.utils.registry import ModuleRegistry


class _Thing:
    def __init__(self, value: int = 0) -> None:
        self.value = value


def test_register_adds_key() -> None:
    reg: ModuleRegistry[_Thing] = ModuleRegistry()

    @reg.register("thing")
    def _load() -> type[_Thing]:
        return _Thing

    assert "thing" in reg.list_available()


def test_get_instantiates_via_lambda() -> None:
    reg: ModuleRegistry[_Thing] = ModuleRegistry()

    @reg.register("thing")
    def _load() -> type[_Thing]:
        return _Thing

    cls = reg.get("thing")
    assert cls is _Thing
    assert isinstance(cls(), _Thing)


def test_duplicate_registration_raises() -> None:
    reg: ModuleRegistry[_Thing] = ModuleRegistry()

    @reg.register("thing")
    def _load() -> type[_Thing]:
        return _Thing

    with pytest.raises(AssertionError):

        @reg.register("thing")
        def _load2() -> type[_Thing]:
            return _Thing


def test_unknown_name_raises_keyerror() -> None:
    reg: ModuleRegistry[_Thing] = ModuleRegistry()
    with pytest.raises(KeyError):
        reg.get("missing")


def test_lazy_load_factory_not_invoked_until_get() -> None:
    reg: ModuleRegistry[_Thing] = ModuleRegistry()
    calls: list[int] = []

    @reg.register("thing")
    def _load() -> type[_Thing]:
        calls.append(1)
        return _Thing

    assert calls == []
    reg.get("thing")
    assert calls == [1]
    reg.get("thing")
    assert calls == [1, 1]


def test_list_available_returns_all_keys() -> None:
    reg: ModuleRegistry[_Thing] = ModuleRegistry()

    @reg.register("a")
    def _a() -> type[_Thing]:
        return _Thing

    @reg.register("b")
    def _b() -> type[_Thing]:
        return _Thing

    assert sorted(reg.list_available()) == ["a", "b"]


def test_labels_default_to_the_name() -> None:
    reg: ModuleRegistry[object] = ModuleRegistry()

    @reg.register("a", "Alpha")
    def _load_a() -> type[object]:
        return object

    @reg.register("b")
    def _load_b() -> type[object]:
        return object

    assert reg.labels() == {"a": "Alpha", "b": "b"}
