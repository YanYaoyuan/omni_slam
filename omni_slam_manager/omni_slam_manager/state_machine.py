"""Pure SLAM lifecycle state machine used by the ROS manager node."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, FrozenSet


class SlamMode(IntEnum):
    STOPPED = 0
    MAPPING = 1
    LOCALIZATION = 2


class SlamState(IntEnum):
    STOPPED = 0
    STARTING_MAPPING = 1
    MAPPING = 2
    SAVING = 3
    MAP_READY = 4
    STARTING_LOCALIZATION = 5
    RELOCALIZING = 6
    LOCALIZED = 7
    DEGRADED = 8
    LOST = 9
    STOPPING = 10
    ERROR = 11


class InvalidTransition(RuntimeError):
    pass


_ALLOWED: Dict[SlamState, FrozenSet[SlamState]] = {
    SlamState.STOPPED: frozenset(
        (SlamState.STARTING_MAPPING, SlamState.STARTING_LOCALIZATION, SlamState.ERROR)
    ),
    SlamState.STARTING_MAPPING: frozenset(
        (SlamState.MAPPING, SlamState.STOPPING, SlamState.ERROR)
    ),
    SlamState.MAPPING: frozenset(
        (SlamState.SAVING, SlamState.STOPPING, SlamState.ERROR)
    ),
    SlamState.SAVING: frozenset(
        (SlamState.MAPPING, SlamState.MAP_READY, SlamState.ERROR)
    ),
    SlamState.MAP_READY: frozenset(
        (
            SlamState.MAPPING,
            SlamState.STOPPING,
            SlamState.STARTING_LOCALIZATION,
            SlamState.ERROR,
        )
    ),
    SlamState.STARTING_LOCALIZATION: frozenset(
        (SlamState.RELOCALIZING, SlamState.STOPPING, SlamState.ERROR)
    ),
    SlamState.RELOCALIZING: frozenset(
        (SlamState.LOCALIZED, SlamState.LOST, SlamState.STOPPING, SlamState.ERROR)
    ),
    SlamState.LOCALIZED: frozenset(
        (SlamState.DEGRADED, SlamState.LOST, SlamState.STOPPING, SlamState.ERROR)
    ),
    SlamState.DEGRADED: frozenset(
        (SlamState.LOCALIZED, SlamState.LOST, SlamState.STOPPING, SlamState.ERROR)
    ),
    SlamState.LOST: frozenset(
        (SlamState.RELOCALIZING, SlamState.STOPPING, SlamState.ERROR)
    ),
    SlamState.STOPPING: frozenset((SlamState.STOPPED, SlamState.ERROR)),
    SlamState.ERROR: frozenset((SlamState.STOPPING, SlamState.STOPPED)),
}


def mode_for_state(state: SlamState) -> SlamMode:
    if state in (
        SlamState.STARTING_MAPPING,
        SlamState.MAPPING,
        SlamState.SAVING,
        SlamState.MAP_READY,
    ):
        return SlamMode.MAPPING
    if state in (
        SlamState.STARTING_LOCALIZATION,
        SlamState.RELOCALIZING,
        SlamState.LOCALIZED,
        SlamState.DEGRADED,
        SlamState.LOST,
    ):
        return SlamMode.LOCALIZATION
    return SlamMode.STOPPED


@dataclass(frozen=True)
class StateSnapshot:
    mode: SlamMode
    state: SlamState
    reason_code: int
    reason_text: str
    sequence: int


class SlamStateMachine:
    def __init__(self) -> None:
        self._state = SlamState.STOPPED
        self._reason_code = 0
        self._reason_text = ""
        self._sequence = 1

    @property
    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            mode=mode_for_state(self._state),
            state=self._state,
            reason_code=self._reason_code,
            reason_text=self._reason_text,
            sequence=self._sequence,
        )

    def transition(
        self, target: SlamState, *, reason_code: int = 0, reason_text: str = ""
    ) -> StateSnapshot:
        if target == self._state:
            return self.snapshot
        if target not in _ALLOWED[self._state]:
            raise InvalidTransition(f"invalid SLAM transition: {self._state.name} -> {target.name}")
        self._state = target
        self._reason_code = reason_code
        self._reason_text = reason_text
        self._sequence += 1
        return self.snapshot

    def fail(self, reason_code: int, reason_text: str) -> StateSnapshot:
        if reason_code <= 0 or not reason_text:
            raise ValueError("ERROR state requires a positive reason code and text")
        if self._state == SlamState.ERROR:
            self._reason_code = reason_code
            self._reason_text = reason_text
            self._sequence += 1
            return self.snapshot
        return self.transition(
            SlamState.ERROR, reason_code=reason_code, reason_text=reason_text
        )
