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
class UsageSnapshot:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

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

    def record(self, usage: Usage) -> None:
        self._requests += 1
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens

    def snapshot(self) -> UsageSnapshot:
        return UsageSnapshot(self._requests, self._input_tokens, self._output_tokens)
