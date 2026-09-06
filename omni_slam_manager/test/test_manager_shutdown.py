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
