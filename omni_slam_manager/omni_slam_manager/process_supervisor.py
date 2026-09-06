"""以独立进程组托管 SLAM 子进程."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
from typing import Dict, List, Optional


class ProcessSupervisorError(RuntimeError):
    pass


def expand_command(command: str, values: Optional[Dict[str, str]] = None) -> List[str]:
    if not command.strip():
        raise ProcessSupervisorError("process command is empty")
    try:
        rendered = command.format(
            **{key: shlex.quote(value) for key, value in (values or {}).items()}
        )
    except KeyError as error:
        raise ProcessSupervisorError(f"unknown command placeholder: {error}") from error
    arguments = shlex.split(rendered)
    if not arguments:
        raise ProcessSupervisorError("process command has no executable")
    return arguments


class ProcessSupervisor:
    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._command: List[str] = []

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def return_code(self) -> Optional[int]:
        return None if self._process is None else self._process.poll()

    @property
    def command(self) -> List[str]:
        return list(self._command)

    def start(self, command: str, values: Optional[Dict[str, str]] = None) -> None:
        if self.running:
            raise ProcessSupervisorError("a SLAM process is already running")
        # 自然退出的旧进程也必须先 wait，避免重复切换模式时积累僵尸进程。
        if self._process is not None:
            self.reap_exit()
        arguments = expand_command(command, values)
        try:
            self._process = subprocess.Popen(arguments, start_new_session=True)
        except OSError as error:
            self._process = None
            raise ProcessSupervisorError(f"failed to start {arguments[0]}: {error}") from error
        self._command = arguments

    @staticmethod
    def _signal_process_group(process: subprocess.Popen, signum: int) -> None:
        """只向本 Supervisor 创建的进程组发信号，不扫描或误杀外部进程."""
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            # 父进程状态与内核进程表之间允许存在一个很短的退出竞态。
            pass
        except OSError as error:
            raise ProcessSupervisorError(
                f"failed to signal SLAM process group {process.pid}: {error}"
            ) from error

    @staticmethod
    def _wait_for_exit(process: subprocess.Popen, timeout_sec: float) -> bool:
        try:
            process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            return False
        return True

    def stop(self, timeout_sec: float = 5.0) -> Optional[int]:
        process = self._process
        if process is None:
            return None
        timeout = max(0.0, float(timeout_sec))
        if process.poll() is None:
            # ROS launch 把 SIGINT 作为正常关闭协议；仅在子进程未按时退出时逐级
            # 升级到 SIGTERM、SIGKILL。每一阶段均作用于完整进程组。
            self._signal_process_group(process, signal.SIGINT)
            if not self._wait_for_exit(process, timeout):
                self._signal_process_group(process, signal.SIGTERM)
                if not self._wait_for_exit(process, timeout):
                    self._signal_process_group(process, signal.SIGKILL)
        return_code = process.wait()
        self._process = None
        self._command = []
        return return_code

    def reap_exit(self) -> Optional[int]:
        process = self._process
        if process is None or process.poll() is None:
            return None
        return_code = process.wait()
        self._process = None
        self._command = []
        return return_code
