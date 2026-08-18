"""Interruptible wait for in-flight ROS service calls.

Kept free of ROS imports so cancel responsiveness is unit-testable off
the robot: the manager polls the completion flag instead of blocking for
the full timeout, and bails as soon as the action goal asks for
cancellation (the ``/map_save`` service cannot itself be aborted once
requested, but the manager must stop acting on its result).
"""

from __future__ import annotations

from typing import Callable
import time


def wait_with_cancel(
    done: Callable[[], bool],
    cancel_requested: Callable[[], bool],
    timeout: float,
    *,
    poll_sec: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> bool:
    """Wait until ``done()`` is true, a cancel is requested, or timeout.

    Returns True when the call completed, False on cancel or timeout.
    ``sleep``/``now`` are injectable so tests can run without real waits.
    """
    if poll_sec <= 0.0:
        raise ValueError("poll_sec must be positive")
    if timeout < 0.0:
        raise ValueError("timeout must be non-negative")
    deadline = now() + timeout
    while not done():
        if cancel_requested():
            return False
        remaining = deadline - now()
        if remaining <= 0.0:
            return False
        sleep(min(poll_sec, remaining))
    return True