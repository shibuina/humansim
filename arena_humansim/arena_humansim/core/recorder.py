from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import yaml
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import WorldGeometry as WorldGeometryMsg
from rclpy.clock import Clock as RclClock
from rclpy.clock import ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import serialize_message
from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata
from rosgraph_msgs.msg import Clock

_GUARDED = ("agent_states", "world_geometry")
_FAST_PERIOD_S = 1.0
_FAST_WINDOW_S = 10.0
_SLOW_PERIOD_S = 30.0


def default_record_dir() -> Path:
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return Path(os.getcwd()) / "recordings" / ts


def _humansim_sha() -> str:
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class BagRecorder:
    """Records the contract topics and writes recording.yaml, the recording manifest, beside the bag."""

    def __init__(
        self,
        node: Node,
        record_dir: Path,
        strict: bool = True,
        on_contamination: Callable[[dict[str, int]], None] | None = None,
        provenance: dict[str, object] | None = None,
    ) -> None:
        self._node = node
        self._record_dir = record_dir
        self._bag_dir = record_dir / "bag"
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._strict = strict
        self._on_contamination = on_contamination

        self._writer = SequentialWriter()
        self._writer.open(
            StorageOptions(uri=str(self._bag_dir), storage_id="mcap"),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )

        ns = node.get_namespace().rstrip("/")
        topics = [
            ("agent_states", "arena_humansim_msgs/msg/AgentStates", AgentStatesMsg, QoSProfile(depth=10)),
            ("world_geometry", "arena_humansim_msgs/msg/WorldGeometry", WorldGeometryMsg, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
            ("/clock", "rosgraph_msgs/msg/Clock", Clock, QoSProfile(depth=10)),
        ]
        self._subs = []
        self._resolved: dict[str, str] = {}
        for i, (topic_name, type_name, msg_type, qos) in enumerate(topics):
            resolved = topic_name if topic_name.startswith("/") else f"{ns}/{topic_name}"
            self._resolved[topic_name] = resolved
            sub = node.create_subscription(msg_type, topic_name, lambda msg, t=resolved: self._write(t, msg), qos)
            self._writer.create_topic(TopicMetadata(id=i, name=resolved, type=type_name, serialization_format="cdr"))
            self._subs.append(sub)

        self._closed = False
        self._first_policies: dict[int, str] | None = None
        self._publisher_counts: dict[str, int] = {}
        self._contaminated = False
        self._manifest = record_dir / "recording.yaml"
        self._provenance: dict[str, object] = {
            "node_fqn": node.get_fully_qualified_name(),
            "pid": os.getpid(),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", ""),
            "namespace": node.get_namespace(),
            "humansim_sha": _humansim_sha(),
            "strict_recording": bool(strict),
            "started_at": _now(),
            **(provenance or {}),
        }
        self._elapsed = 0.0
        self._check_publishers()
        # the guard windows are wall-clock seconds
        self._timer = node.create_timer(_FAST_PERIOD_S, self._on_timer, clock=RclClock(clock_type=ClockType.STEADY_TIME))
        self._write_manifest()
        node.get_logger().info(f"BagRecorder writing to {self._bag_dir}")

    def _check_publishers(self) -> bool:
        clean = True
        for topic in _GUARDED:
            resolved = self._resolved[topic]
            count = int(self._node.count_publishers(resolved))
            self._publisher_counts[resolved] = count
            if count > 1:
                clean = False
        if not clean and not self._contaminated:
            self._contaminated = True
            self._node.get_logger().error(
                f"recording contaminated, publisher counts {self._publisher_counts} on {self._provenance['node_fqn']} "
                f"(ROS_DOMAIN_ID={self._provenance['ros_domain_id']}): a second simulator is publishing onto this bus"
            )
            self._write_manifest()
            if self._strict and self._on_contamination is not None:
                self._on_contamination(dict(self._publisher_counts))
        return clean

    def _on_timer(self) -> None:
        self._elapsed += _FAST_PERIOD_S
        if self._elapsed <= _FAST_WINDOW_S or abs(self._elapsed % _SLOW_PERIOD_S) < _FAST_PERIOD_S / 2:
            self._check_publishers()

    def _write_manifest(self) -> None:
        doc = {
            **self._provenance,
            "contaminated": bool(self._contaminated),
            "publishers": dict(self._publisher_counts),
            "first_message_policies": self._first_policies or {},
        }
        try:
            self._manifest.write_text(yaml.safe_dump(doc, sort_keys=False))
        except OSError as exc:
            self._node.get_logger().warning(f"could not write {self._manifest}: {exc}")

    def _write(self, topic_name: str, msg: AgentStatesMsg | WorldGeometryMsg | Clock) -> None:
        if self._closed:
            return
        if isinstance(msg, Clock):
            t = self._node.get_clock().now().nanoseconds
        else:
            t = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            if self._first_policies is None and isinstance(msg, AgentStatesMsg):
                self._first_policies = {int(a.agent_id): str(a.policy) for a in msg.agents}
                self._write_manifest()
        self._writer.write(topic_name, serialize_message(msg), t)

    @property
    def record_dir(self) -> Path:
        return self._record_dir

    @property
    def contaminated(self) -> bool:
        return self._contaminated

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._node.destroy_timer(self._timer)
        except Exception:
            pass
        self._check_publishers()
        self._provenance["finished_at"] = _now()
        self._write_manifest()
        for sub in self._subs:
            try:
                self._node.destroy_subscription(sub)
            except Exception:
                pass
        del self._writer
