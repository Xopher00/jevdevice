"""Token/request accounting for JevClient.ask() calls -- nothing tracked this before,
so a runaway loop (a chain of chained actions, a batch eval) had no visible cost signal.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclasses.dataclass(frozen=True)
class EngineInfo:
    """Which engine answered, on exactly which checkpoint -- the A/B join key
    and the calibration anchor. Snapshots from both engines must distinguish
    themselves; the Jev path records this once at construction, the Laya path
    refreshes it per ask with the predict's own routing payload."""

    engine: str
    model_revision: str
    model: str | None = None  # laya: "typed-decisions"; jev: the model IS the revision
    routing: dict | None = None  # laya's per-predict RouteDecision payload


@dataclasses.dataclass(frozen=True)
class UsageSnapshot:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    engine_info: EngineInfo | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageLedger:
    """Accumulates real per-request usage. One instance per JevClient; asyncio-single-threaded,
    no lock needed (jevdevice never shares a JevClient across concurrent event loops)."""

    def __init__(self) -> None:
        self._requests = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._engine_info: EngineInfo | None = None

    def record(self, usage: Usage) -> None:
        self._requests += 1
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens

    def record_engine(self, info: EngineInfo) -> None:
        self._engine_info = info

    @property
    def engine_info(self) -> EngineInfo | None:
        return self._engine_info

    def snapshot(self) -> UsageSnapshot:
        return UsageSnapshot(self._requests, self._input_tokens, self._output_tokens, self._engine_info)
