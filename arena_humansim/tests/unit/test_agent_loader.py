from __future__ import annotations

import logging
from pathlib import Path

import attrs
import pytest
from arena_humansim.core.agents import loader
from arena_humansim.core.agents.types import AgentType


def test_get_share_agent_types_dir_fallback_or_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import ament_index_python.packages as aip

    def _raise(_name: str) -> str:
        raise RuntimeError("no ament")

    monkeypatch.setattr(aip, "get_package_share_directory", _raise)

    result = loader._get_share_agent_types_dir()
    if result is not None:
        assert result.is_dir()
        assert result.name == "agent_types"
    else:
        assert result is None


def test_load_agent_type_raw_fills_name_from_stem_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "wanderer.yaml"
    path.write_text("mode: simple\n")

    name, raw, src = loader.load_agent_type_raw_from_file(path)

    assert name == "wanderer"
    assert raw["name"] == "wanderer"
    assert raw["mode"] == "simple"
    assert src == path.resolve()


def test_load_agent_type_raw_empty_yaml_yields_empty_dict(tmp_path: Path) -> None:
    path = tmp_path / "blank.yaml"
    path.write_text("")

    name, raw, src = loader.load_agent_type_raw_from_file(path)

    assert name == "blank"
    assert raw == {"name": "blank"}
    assert src == path.resolve()


def test_load_agent_types_raw_from_dir_skips_bad_yamls(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text("name: good\nmode: simple\n")
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nmode: [unterminated\n")

    result = loader.load_agent_types_raw_from_dir(tmp_path)

    assert "good" in result
    assert "bad" not in result
    assert result["good"][1] == good.resolve()


def test_load_agent_types_raw_from_dir_missing_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert loader.load_agent_types_raw_from_dir(missing) == {}


def test_load_agent_types_empty_when_share_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "_get_share_agent_types_dir", lambda: None)
    assert loader.load_agent_types() == {}
    assert loader._load_default_agent_types_raw() == {}


def test_resolve_agent_type_by_name_path_and_registry(tmp_path: Path) -> None:
    path = tmp_path / "walker.yaml"
    path.write_text("name: walker\nmode: simple\n")

    by_path = loader.resolve_agent_type_name(str(path), registry={})
    assert isinstance(by_path, AgentType)
    assert by_path.name == "walker"

    missing = loader.resolve_agent_type_name(str(tmp_path / "nope.yaml"), registry={})
    assert missing is None

    dummy = AgentType(name="adult")
    by_registry = loader.resolve_agent_type_name("adult", registry={"adult": dummy})
    assert by_registry is dummy

    assert loader.resolve_agent_type_name("unknown", registry={}) is None


def test_load_agent_types_with_extends_resolves_inline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    defaults_dir = tmp_path / "defaults"
    defaults_dir.mkdir()
    parent_yaml = defaults_dir / "parent.yaml"
    parent_yaml.write_text("name: parent\nmode: simple\ndesired_velocity:\n  mean: 2.0\nagent_radius:\n  mean: 0.5\n")

    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    child_yaml = scenario_dir / "child.yaml"
    child_yaml.write_text("name: child\nextends: parent\ndesired_velocity:\n  mean: 3.3\n")

    monkeypatch.setattr(loader, "_get_share_agent_types_dir", lambda: defaults_dir)

    result = loader.load_agent_types(scenario_dir=scenario_dir)

    assert set(result) == {"parent", "child"}
    assert result["child"].desired_velocity.mean == 3.3
    assert result["child"].agent_radius.mean == 0.5
    assert result["child"].extends is None
    assert result["parent"].desired_velocity.mean == 2.0


def test_load_agent_types_without_extends_returns_structured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    defaults_dir = tmp_path / "defaults"
    defaults_dir.mkdir()
    (defaults_dir / "plain.yaml").write_text("name: plain\nmode: simple\n")

    monkeypatch.setattr(loader, "_get_share_agent_types_dir", lambda: defaults_dir)

    result = loader.load_agent_types()
    assert "plain" in result
    assert isinstance(result["plain"], AgentType)
    assert result["plain"].name == "plain"


def _load_dir_with_warnings(directory: Path, caplog: pytest.LogCaptureFixture) -> tuple[set[str], list[str]]:
    """Names loaded from the directory and the loader's warning messages, with the handler on the logger itself since it may not propagate."""
    loader._log.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=loader._log.name):
            result = loader.load_agent_types_raw_from_dir(directory)
    finally:
        loader._log.removeHandler(caplog.handler)
    records = {id(r): r for r in caplog.records if r.name == loader._log.name}
    return set(result), [r.getMessage() for r in records.values()]


def test_load_agent_types_raw_from_dir_warns_and_skips_unknown_locomotion_key(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "good.yaml").write_text("name: good\nlocomotion: {kinematics: along_heading}\n")
    (tmp_path / "typo.yaml").write_text("name: typo\nlocomotion: {kinematix: along_heading}\n")

    names, warnings = _load_dir_with_warnings(tmp_path, caplog)

    assert names == {"good"}
    assert len(warnings) == 1
    assert "typo.yaml" in warnings[0]
    assert "kinematix" in warnings[0]


def test_load_agent_types_raw_from_dir_skips_scenario_files_silently(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "good.yaml").write_text("name: good\n")
    (tmp_path / "scene.yaml").write_text("name: scene\ndescription: two rooms\nsimulation: {seed: 1}\nmodules: {local_planner: hsfm}\nagents: []\nwalls: []\n")
    for key in loader.SCENARIO_ONLY_KEYS:
        (tmp_path / f"only_{key}.yaml").write_text(f"name: only_{key}\n{key}: null\n")

    names, warnings = _load_dir_with_warnings(tmp_path, caplog)

    assert names == {"good"}
    assert warnings == []


def test_scenario_only_keys_are_the_scenario_fields_an_agent_type_lacks() -> None:
    from arena_humansim.utils.scenario import ScenarioConfig

    scenario = {f.name for f in attrs.fields(ScenarioConfig)}
    agent_type = {f.name for f in attrs.fields(AgentType)}
    assert set(loader.SCENARIO_ONLY_KEYS) == scenario - agent_type
    assert len(set(loader.SCENARIO_ONLY_KEYS)) == len(loader.SCENARIO_ONLY_KEYS)


def test_load_agent_type_from_file_extending_adult_with_locomotion(tmp_path: Path) -> None:
    path = tmp_path / "cart.yaml"
    path.write_text("name: cart\nextends: adult\nlocomotion:\n  cadence: {max: 1.2}\n")

    at = loader.load_agent_type_from_file(path)

    assert at.locomotion_active is True
    assert at.locomotion.cadence.max.mean == 1.2
    assert at.locomotion.cadence.base.mean == 0.4
    assert at.desired_velocity.mean == 1.1
