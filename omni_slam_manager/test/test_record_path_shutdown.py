"""验证路径录制器只忽略 ROS context 关闭导致的 WaitSet 竞态。"""

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "global_path_tools"
    / "record_path.py"
)
SPEC = importlib.util.spec_from_file_location("omni_record_path", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
record_path = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(record_path)


def _raise(error: Exception):
    raise error


def test_rcl_error_after_context_shutdown_is_clean(monkeypatch) -> None:
    error = record_path.rclpy_implementation.RCLError("context is not valid")
    monkeypatch.setattr(record_path.rclpy, "ok", lambda: False)
    monkeypatch.setattr(
        record_path.rclpy,
        "spin_once",
        lambda *_args, **_kwargs: _raise(error),
    )

    assert not record_path._spin_once_until_context_shutdown(object(), 0.1)


def test_rcl_error_with_valid_context_is_not_hidden(monkeypatch) -> None:
    error = record_path.rclpy_implementation.RCLError("unexpected failure")
    monkeypatch.setattr(record_path.rclpy, "ok", lambda: True)
    monkeypatch.setattr(
        record_path.rclpy,
        "spin_once",
        lambda *_args, **_kwargs: _raise(error),
    )

    with pytest.raises(record_path.rclpy_implementation.RCLError):
        record_path._spin_once_until_context_shutdown(object(), 0.1)
