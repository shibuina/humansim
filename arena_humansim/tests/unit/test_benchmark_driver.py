from __future__ import annotations

import threading

import pytest

rclpy = pytest.importorskip("rclpy")

from arena_humansim.utils.benchmark import BenchmarkDriver
from arena_humansim_msgs.msg import AgentFrame as AgentFrameMsg
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from tests.unit._bags import Agent, frame_msg


def test_driver_times_ticks_from_agent_frames(rclpy_context) -> None:
    if rclpy_context is None:
        pytest.skip("rclpy unavailable")
    driver = BenchmarkDriver()
    engine = Node("benchmark_fake_engine")
    pub = engine.create_publisher(AgentFrameMsg, "agent_states", 10)
    frame = frame_msg(0, [Agent(1, 0.0, 0.0)], ["sfm"])
    timer = engine.create_timer(0.02, lambda: pub.publish(frame))
    executor = MultiThreadedExecutor()
    executor.add_node(driver)
    executor.add_node(engine)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        ticks = driver.collect_ticks(n_ticks=5, warmup=2)
    finally:
        executor.shutdown()
        spinner.join(timeout=5.0)
        engine.destroy_timer(timer)
        engine.destroy_node()
        driver.destroy_node()

    assert len(ticks) == 5
    assert all(t > 0.0 for t in ticks)
