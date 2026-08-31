"""Pure SLAM telemetry quality tracking (pose jumps + ICP fitness).

The manager keeps this logic in a ROS-free module so the quality contract
is unit-testable off the robot:

* A "pose jump" is a discontinuity in the global odometry stream -- the
  classic symptom of a bad relocalization or an odom reset.  It is counted
  in :class:`PoseJumpTracker` and surfaced as ``SlamStatus.pose_jump_count``.
* The ICP fitness value is the alignment score published by the
  relocalization node (lower is better).  :class:`FitnessTracker` remembers
  the newest finite sample; it stays NaN while no source exists, e.g. in
  mapping mode or before the first ICP iteration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PoseSample:
    """One global odometry pose, timestamped with steady-clock seconds."""

    x: float
    y: float
    z: float
    yaw: float
    stamp: float


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract the z-axis heading from a quaternion (w, x, y, z)."""
    sin_yaw_cos_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw_cos_yaw, cos_yaw)


def _angle_delta(angle: float) -> float:
    """Wrap a signed angle delta to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PoseJumpTracker:
    """Count discontinuities in the global odometry stream.

    A jump is a displacement of ``position_threshold_m`` or more, or a
    heading change of ``yaw_threshold_rad`` or more, relative to the
    previous sample.  One discontinuity counts once: a refractory window
    (``refractory_sec``) suppresses re-counting the same event across
    consecutive frames.  Non-finite samples are ignored and never become
    the comparison baseline.
    """

    def __init__(
        self,
        position_threshold_m: float = 0.5,
        yaw_threshold_rad: float = math.radians(45.0),
        refractory_sec: float = 0.5,
    ) -> None:
        if position_threshold_m <= 0.0:
            raise ValueError("position_threshold_m must be positive")
        if yaw_threshold_rad <= 0.0:
            raise ValueError("yaw_threshold_rad must be positive")
        if refractory_sec < 0.0:
            raise ValueError("refractory_sec must be non-negative")
        self._position_threshold_m = position_threshold_m
        self._yaw_threshold_rad = yaw_threshold_rad
        self._refractory_sec = refractory_sec
        self._last_sample: Optional[PoseSample] = None
        self._last_jump_at: float = -math.inf
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._last_sample = None
        self._last_jump_at = -math.inf
        self._count = 0

    def observe(self, sample: PoseSample) -> bool:
        """Ingest one pose; return True when a jump was counted."""
        values = (sample.x, sample.y, sample.z, sample.yaw, sample.stamp)
        if not all(math.isfinite(value) for value in values):
            return False
        previous = self._last_sample
        self._last_sample = sample
        if previous is None:
            return False
        dx = sample.x - previous.x
        dy = sample.y - previous.y
        dz = sample.z - previous.z
        displacement = math.sqrt(dx * dx + dy * dy + dz * dz)
        heading_change = abs(_angle_delta(sample.yaw - previous.yaw))
        if (
            displacement < self._position_threshold_m
            and heading_change < self._yaw_threshold_rad
        ):
            return False
        if sample.stamp - self._last_jump_at < self._refractory_sec:
            return False
        self._last_jump_at = sample.stamp
        self._count += 1
        return True


class FitnessTracker:
    """Remember the newest finite ICP fitness score (NaN until then)."""

    def __init__(self) -> None:
        self._value: float = math.nan

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = math.nan

    def observe(self, value: float) -> None:
        if math.isfinite(value):
            self._value = value