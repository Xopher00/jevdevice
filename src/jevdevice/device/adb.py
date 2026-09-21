"""The Android phone family: AdbDevice over the frozen AdbTransport."""

from __future__ import annotations

from jevdevice.transport import AdbTransport, RunResult

from .protocol import DEFAULT_COMMAND_TIMEOUT_S


class AdbDevice:
    """The real Android device as a Device: a thin wrapper around the frozen
    AdbTransport (wrap, don't rewrite). Everything device-specific below the
    protocol stays in transport.py."""

    def __init__(self, transport: AdbTransport) -> None:
        self._transport = transport
        # The journal's device column: one identifier per connected device,
        # stable across sessions. `serial` stays as the historical attribute
        # name (existing callers read it); `name` is the protocol's
        # family-agnostic member -- for Android they are the same string.
        self.serial = transport.serial
        self.name = self.serial

    async def dump_hierarchy(self) -> str:
        return await self._transport.dump_hierarchy()

    async def run(self, command: str, timeout: float = DEFAULT_COMMAND_TIMEOUT_S) -> RunResult:
        return await self._transport.run(command, timeout)

    async def run_binary(self, command: str, timeout: float = DEFAULT_COMMAND_TIMEOUT_S) -> bytes:
        return await self._transport.run_binary(command, timeout)

    async def window_size(self) -> tuple[int, int]:
        return await self._transport.window_size()
