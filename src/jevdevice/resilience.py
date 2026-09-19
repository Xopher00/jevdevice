"""Circuit breaker for a LOOP of live jev.ask() calls (a chained sequence of actions),
distinct from jev.py's own per-request 429/529 retry: that covers one request, this covers
a run of many requests hitting a genuinely dead/unreachable endpoint. Opt-in, not wired into
JevClient -- a single CLI/MCP call has nothing to gain from failing fast after N failures
when N is 1.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Callable


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    def __init__(self, message: str, *, remaining_seconds: float) -> None:
        super().__init__(message)
        self.remaining_seconds = remaining_seconds


@dataclass
class CircuitBreaker:
    """After `failure_threshold` consecutive failures, fails fast for `cooldown_seconds`,
    then allows exactly one probe through. A probe success closes the circuit; a probe
    failure reopens it with a fresh cooldown. Never sleeps -- the caller decides whether
    to wait out the cooldown."""

    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        return self._state

    def _check(self) -> None:
        if self._state is CircuitState.CLOSED:
            return
        now = self.clock()
        if self._state is CircuitState.OPEN and now >= self._opened_at + self.cooldown_seconds:
            self._state = CircuitState.HALF_OPEN
            return
        remaining = 0.0 if self._state is CircuitState.HALF_OPEN else max(0.0, self._opened_at + self.cooldown_seconds - now)
        raise CircuitOpenError(f"circuit {self._state.value}: {remaining:.1f}s remaining", remaining_seconds=remaining)

    def record_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._state is CircuitState.HALF_OPEN or self._consecutive_failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self.clock()

    async def call(self, fn, *args, **kwargs):
        self._check()
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result
