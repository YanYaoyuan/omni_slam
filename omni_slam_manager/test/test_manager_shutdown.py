"""验证 SLAM Manager 仅忽略 ROS context 关闭引发的退出异常。"""

import pytest
from rclpy.executors import ExternalShutdownException

from omni_slam_manager import manager_node


class _FailingExecutor:
    """用于精确注入执行器异常的最小测试替身。"""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def spin(self) -> None:
        raise self._error


def test_external_shutdown_is_a_clean_exit() -> None:
    executor = _FailingExecutor(ExternalShutdownException())

    manager_node._spin_until_context_shutdown(executor)


def test_rcl_error_after_context_shutdown_is_a_clean_exit(monkeypatch) -> None:
    monkeypatch.setattr(manager_node.rclpy, "ok", lambda: False)
    error = manager_node.rclpy_implementation.RCLError("context is not valid")
    executor = _FailingExecutor(error)

    manager_node._spin_until_context_shutdown(executor)


def test_rcl_error_while_context_is_valid_is_not_hidden(monkeypatch) -> None:
    monkeypatch.setattr(manager_node.rclpy, "ok", lambda: True)
    error = manager_node.rclpy_implementation.RCLError("unexpected wait-set failure")
    executor = _FailingExecutor(error)

    with pytest.raises(manager_node.rclpy_implementation.RCLError):
        manager_node._spin_until_context_shutdown(executor)


def test_main_takes_instance_lock_before_initializing_ros(monkeypatch) -> None:
    events = []

    class FakeLock:
        def __init__(self, _path: str) -> None:
            events.append("lock-created")

        def __enter__(self):
            events.append("lock-acquired")
            return self

        def __exit__(self, *_args) -> None:
            events.append("lock-released")

    class FakeNode:
        def __init__(self) -> None:
            events.append("node-created")

        def destroy_node(self) -> None:
            events.append("node-destroyed")

    class FakeExecutor:
        def __init__(self, *, num_threads: int) -> None:
            assert num_threads == 4
            events.append("executor-created")

        def add_node(self, _node) -> None:
            events.append("node-added")

        def shutdown(self) -> None:
            events.append("executor-stopped")

    monkeypatch.setattr(manager_node, "InstanceLock", FakeLock)
    monkeypatch.setattr(manager_node, "SlamManagerNode", FakeNode)
    monkeypatch.setattr(manager_node, "MultiThreadedExecutor", FakeExecutor)
    monkeypatch.setattr(manager_node, "_spin_until_context_shutdown", lambda _e: None)
    monkeypatch.setattr(manager_node.rclpy, "init", lambda args=None: events.append("ros-init"))
    monkeypatch.setattr(manager_node.rclpy, "ok", lambda: True)
    monkeypatch.setattr(manager_node.rclpy, "shutdown", lambda: events.append("ros-stop"))

    manager_node.main([])

    assert events.index("lock-acquired") < events.index("ros-init")
    assert events.index("ros-init") < events.index("node-created")
    assert events[-1] == "lock-released"


def test_main_cleans_ros_context_when_node_construction_fails(monkeypatch) -> None:
    events = []

    class FakeLock:
        def __init__(self, _path: str) -> None:
            pass

        def __enter__(self):
            events.append("lock-acquired")
            return self

        def __exit__(self, *_args) -> None:
            events.append("lock-released")

    def fail_node_construction():
        raise RuntimeError("node construction failed")

    monkeypatch.setattr(manager_node, "InstanceLock", FakeLock)
    monkeypatch.setattr(manager_node, "SlamManagerNode", fail_node_construction)
    monkeypatch.setattr(manager_node.rclpy, "init", lambda args=None: events.append("ros-init"))
    monkeypatch.setattr(manager_node.rclpy, "ok", lambda: True)
    monkeypatch.setattr(manager_node.rclpy, "shutdown", lambda: events.append("ros-stop"))

    with pytest.raises(RuntimeError, match="node construction failed"):
        manager_node.main([])

    assert events == ["lock-acquired", "ros-init", "ros-stop", "lock-released"]
