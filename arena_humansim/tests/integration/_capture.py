from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

LATCHED = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
STREAM = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)


def subscribe(node: Node, msg_type: Any, topic: str, qos: QoSProfile) -> list[Any]:
    received: list[Any] = []
    node.create_subscription(msg_type, topic, received.append, qos)
    return received


def spin_until(executor: SingleThreadedExecutor, done: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    assert done(), f"condition not met within {timeout}s"


def settle(executor: SingleThreadedExecutor, duration: float = 0.3) -> None:
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
