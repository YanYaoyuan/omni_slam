"""SLAM Manager 宿主机单实例锁."""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
import stat
from typing import Optional


class InstanceLockError(RuntimeError):
    """同一宿主机上已有 SLAM Manager owner 时抛出."""


class InstanceLock:
    """
    使用内核 ``flock`` 持有进程生命周期级独占锁.

    PID 文本只供诊断，真正的所有权由仍然打开的文件描述符和内核锁证明。这样即使
    进程异常退出，内核也会自动释放锁，不会遗留需要猜测有效性的 PID 文件。
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path).expanduser()
        self._fd: Optional[int] = None

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> None:
        if self._fd is not None:
            return
        if not self._path.is_absolute():
            raise InstanceLockError("instance lock path must be absolute")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        except OSError as error:
            raise InstanceLockError(
                f"cannot create instance lock directory {self._path.parent}: {error}"
            ) from error
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._path, flags, 0o640)
        except OSError as error:
            raise InstanceLockError(
                f"cannot open instance lock {self._path}: {error}"
            ) from error
        try:
            # 即使平台没有 O_CLOEXEC，也不得让 FAST-LIO/ICP 子进程继承 owner 锁。
            os.set_inheritable(descriptor, False)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise InstanceLockError(
                    f"instance lock is not a regular file: {self._path}"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EACCES, errno.EAGAIN):
                    raise InstanceLockError(
                        f"another omni_slam_manager owns {self._path}"
                    ) from error
                raise InstanceLockError(
                    f"cannot lock instance file {self._path}: {error}"
                ) from error

            os.ftruncate(descriptor, 0)
            payload = f"{os.getpid()}\n".encode("ascii")
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
            self._fd = descriptor
        except Exception:
            os.close(descriptor)
            raise

    def release(self) -> None:
        descriptor = self._fd
        self._fd = None
        if descriptor is not None:
            os.close(descriptor)

    def __enter__(self) -> "InstanceLock":
        self.acquire()
        return self

    def __exit__(self, _error_type, _error, _traceback) -> None:
        self.release()
