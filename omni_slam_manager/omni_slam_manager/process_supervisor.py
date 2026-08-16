"""Small subprocess supervisor with process-group shutdown semantics."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
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
        arguments = expand_command(command, values)
        try:
            self._process = subprocess.Popen(arguments, start_new_session=True)
        except OSError as error:
            self._process = None
            raise ProcessSupervisorError(f"failed to start {arguments[0]}: {error}") from error
        self._command = arguments

    def stop(self, timeout_sec: float = 5.0) -> Optional[int]:
        process = self._process
        if process is None:
            return None
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + max(0.0, timeout_sec)
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
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
