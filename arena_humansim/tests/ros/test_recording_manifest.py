"""The recording manifest written beside the bag."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from rclpy.node import Node
from rclpy.parameter import Parameter

from arena_humansim.core.recorder import BagRecorder
from tests.ros._helpers import make_system

pytestmark = pytest.mark.ros


def _params(**kwargs: object) -> list[Parameter]:
    return [Parameter(k, value=v) for k, v in kwargs.items()]


def _manifest(record_dir: Path) -> dict:
    return yaml.safe_load((record_dir / "recording.yaml").read_text())


def test_manifest_records_provenance_and_the_clean_close(tmp_path: Path) -> None:
    sys_ = make_system(
        client_node_name="manifest_client_clean",
        parameter_overrides=_params(record_bag=True, record_dir=str(tmp_path), strict_recording=False, trial_id="t1"),
    )
    try:
        sys_.drain(0.5)
        manifest = _manifest(tmp_path)
        assert manifest["trial_id"] == "t1"
        assert manifest["contaminated"] is False
        assert "finished_at" not in manifest
    finally:
        sys_.shutdown()
    manifest = _manifest(tmp_path)
    assert manifest["finished_at"]
    assert manifest["contaminated"] is False


def test_second_publisher_marks_the_manifest_contaminated(tmp_path: Path) -> None:
    sys_ = make_system(
        client_node_name="manifest_client_dirty",
        parameter_overrides=_params(record_bag=True, record_dir=str(tmp_path), strict_recording=False),
    )
    intruder = sys_.client_node.create_publisher(AgentStatesMsg, "agent_states", 10)
    try:
        sys_.drain(1.0)
        assert sys_.manager._recorder._check_publishers() is False
        manifest = _manifest(tmp_path)
        assert manifest["contaminated"] is True
        assert manifest["publishers"]["/agent_states"] == 2
    finally:
        sys_.client_node.destroy_publisher(intruder)
        sys_.shutdown()


def test_strict_guard_calls_back_with_the_counts(tmp_path: Path) -> None:
    node = Node("manifest_strict_node")
    publishers = [node.create_publisher(AgentStatesMsg, "agent_states", 10) for _ in range(2)]
    seen: list[dict[str, int]] = []
    try:
        recorder = BagRecorder(node, tmp_path, strict=True, on_contamination=seen.append)
        assert seen == [{"/agent_states": 2, "/world_geometry": 0}]
        assert recorder.contaminated is True
        recorder.close()
        assert seen == [{"/agent_states": 2, "/world_geometry": 0}]
        assert _manifest(tmp_path)["contaminated"] is True
    finally:
        for pub in publishers:
            node.destroy_publisher(pub)
        node.destroy_node()
