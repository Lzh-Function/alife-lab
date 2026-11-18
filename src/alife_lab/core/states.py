"""Run state machine (spec section 2.3)."""

from __future__ import annotations

from enum import StrEnum


class RunState(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


TERMINAL_STATES = frozenset({RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED})
ACTIVE_STATES = frozenset(
    {
        RunState.STARTING,
        RunState.RUNNING,
        RunState.PAUSING,
        RunState.PAUSED,
        RunState.RESUMING,
    }
)
# States from which a new worker job may be started from a checkpoint.
RESTORABLE_STATES = frozenset({RunState.INTERRUPTED, RunState.FAILED})

_S = RunState
ALLOWED_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    _S.PENDING: frozenset({_S.QUEUED, _S.CANCELLED}),
    _S.QUEUED: frozenset({_S.STARTING, _S.CANCELLED, _S.INTERRUPTED}),
    _S.STARTING: frozenset({_S.RUNNING, _S.PAUSED, _S.FAILED, _S.CANCELLED, _S.INTERRUPTED}),
    _S.RUNNING: frozenset(
        {_S.PAUSING, _S.PAUSED, _S.COMPLETED, _S.FAILED, _S.CANCELLED, _S.INTERRUPTED}
    ),
    _S.PAUSING: frozenset({_S.PAUSED, _S.COMPLETED, _S.FAILED, _S.CANCELLED, _S.INTERRUPTED}),
    _S.PAUSED: frozenset(
        {_S.RESUMING, _S.RUNNING, _S.COMPLETED, _S.FAILED, _S.CANCELLED, _S.INTERRUPTED}
    ),
    _S.RESUMING: frozenset(
        {_S.RUNNING, _S.PAUSED, _S.COMPLETED, _S.FAILED, _S.CANCELLED, _S.INTERRUPTED}
    ),
    _S.INTERRUPTED: frozenset({_S.QUEUED, _S.CANCELLED}),
    _S.FAILED: frozenset({_S.QUEUED}),
    _S.COMPLETED: frozenset(),
    _S.CANCELLED: frozenset(),
}


class InvalidTransition(ValueError):
    def __init__(self, current: RunState, target: RunState) -> None:
        super().__init__(f"invalid run state transition {current.value} -> {target.value}")
        self.current = current
        self.target = target


def check_transition(current: RunState, target: RunState) -> None:
    if current == target:
        return
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransition(current, target)
