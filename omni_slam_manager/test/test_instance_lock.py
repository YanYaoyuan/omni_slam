"""SLAM Manager 单实例锁回归测试."""

from pathlib import Path
import os
import tempfile

import pytest

from omni_slam_manager.instance_lock import InstanceLock, InstanceLockError


def test_second_owner_is_rejected_and_release_allows_reacquire() -> None:
    with tempfile.TemporaryDirectory() as directory:
        lock_path = str(Path(directory) / "slam-manager.lock")
        first = InstanceLock(lock_path)
        second = InstanceLock(lock_path)

        first.acquire()
        assert Path(lock_path).read_text(encoding="ascii") == f"{os.getpid()}\n"
        assert first._fd is not None
        assert not os.get_inheritable(first._fd)
        with pytest.raises(InstanceLockError, match="another omni_slam_manager"):
            second.acquire()

        first.release()
        second.acquire()
        second.release()


def test_relative_lock_path_is_rejected() -> None:
    with pytest.raises(InstanceLockError, match="must be absolute"):
        InstanceLock("relative.lock").acquire()


def test_context_manager_releases_lock_when_body_raises() -> None:
    with tempfile.TemporaryDirectory() as directory:
        lock_path = str(Path(directory) / "slam-manager.lock")
        with pytest.raises(RuntimeError, match="injected"):
            with InstanceLock(lock_path):
                raise RuntimeError("injected")

        with InstanceLock(lock_path):
            pass
