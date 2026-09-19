"""Transport: the only thing hardwired per device type.

Everything else — what to probe, how to read the output, what command performs
an action, how to verify it — is generated per goal, not written in advance.
A new device type means one new Transport, nothing else.
"""

from __future__ import annotations

import asyncio
import platform
from dataclasses import dataclass
from typing import Protocol


@dataclass
class RunResult:
    stdout: str
    stderr: str
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class Transport(Protocol):
    async def describe(self) -> str:
        """A short device fingerprint, e.g. 'Android 14, Samsung SM-S921B'."""
        ...

    async def run(self, command: str, timeout: float = 15.0) -> RunResult: ...


class AdbTransport:
    def __init__(self, serial: str, adb_command: str = "adb") -> None:
        self.serial = serial
        self._adb = adb_command
        self._u2 = None  # lazy: only paid if dump_hierarchy is actually called

    def _u2_device(self):
        if self._u2 is None:
            import uiautomator2

            self._u2 = uiautomator2.connect(self.serial)
        return self._u2

    async def dump_hierarchy(self) -> str:
        """A real uiautomator2 companion server sideloaded once and kept resident on-device
        answers this in ~0.2-0.3s; a fresh `uiautomator dump` process cold-starts in ~2.2s
        every call (confirmed live) because it reloads the ART runtime each time."""
        return await asyncio.to_thread(self._u2_device().dump_hierarchy)

    async def window_size(self) -> tuple[int, int]:
        return await asyncio.to_thread(self._u2_device().window_size)

    async def describe(self) -> str:
        return f"adb:{self.serial}"  # an identifier only, not a real device fingerprint

    async def run(self, command: str, timeout: float = 15.0) -> RunResult:
        # stdin=DEVNULL: without it, a subprocess inherits the real terminal's stdin by
        # default, so a device probe can silently consume input meant for a later approval prompt.
        process = await asyncio.create_subprocess_exec(
            self._adb, "-s", self.serial, "shell", command,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            return RunResult(stdout="", stderr="timed out", exit_code=124)
        return RunResult(stdout=stdout.decode(errors="replace"), stderr=stderr.decode(errors="replace"), exit_code=process.returncode or 0)

    async def run_binary(self, command: str, timeout: float = 15.0) -> bytes:
        """`exec-out` streams raw stdout bytes -- for commands like `screencap -p`
        whose output would be corrupted by `run`'s text decoding."""
        process = await asyncio.create_subprocess_exec(
            self._adb, "-s", self.serial, "exec-out", command,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return stdout


class LocalShellTransport:
    async def describe(self) -> str:
        return platform.platform()  # stdlib, not a shell call

    async def run(self, command: str, timeout: float = 15.0) -> RunResult:
        process = await asyncio.create_subprocess_shell(
            command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            return RunResult(stdout="", stderr="timed out", exit_code=124)
        return RunResult(stdout=stdout.decode(errors="replace"), stderr=stderr.decode(errors="replace"), exit_code=process.returncode or 0)
