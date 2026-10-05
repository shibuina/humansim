from __future__ import annotations

import signal
import subprocess

import pytest

pytest.importorskip("rclpy")

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_prefix
from arena_humansim_msgs.msg import AgentViz as AgentVizMsg
from rclpy.executors import SingleThreadedExecutor
from visualization_msgs.msg import MarkerArray

from ._capture import STREAM, settle, spin_until

NS = "/viz_node_subscription"


def test_viz_node_subscribes_only_while_its_output_is_watched(rclpy_context: object) -> None:  # noqa: ARG001
    try:
        prefix = get_package_prefix("arena_humansim_viz")
    except PackageNotFoundError:
        pytest.skip("arena_humansim_viz is not built")
    proc = subprocess.Popen([f"{prefix}/lib/arena_humansim_viz/arena_humansim_viz_node", "--ros-args", "-r", f"__ns:={NS}"])
    node = rclpy.create_node("viz_node_subscription_probe")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        state = node.create_publisher(AgentVizMsg, f"{NS}/viz_state", STREAM)
        spin_until(executor, lambda: node.count_publishers(f"{NS}/viz") == 1, timeout=20.0)
        settle(executor, 2.5)
        assert state.get_subscription_count() == 0

        viewer = node.create_subscription(MarkerArray, f"{NS}/viz", lambda _msg: None, 100)
        spin_until(executor, lambda: state.get_subscription_count() == 1, timeout=10.0)

        node.destroy_subscription(viewer)
        spin_until(executor, lambda: state.get_subscription_count() == 0, timeout=10.0)
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10.0)
        executor.remove_node(node)
        node.destroy_node()
