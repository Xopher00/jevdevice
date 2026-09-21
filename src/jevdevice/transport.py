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


async def _communicate_or_kill(
    process: asyncio.subprocess.Process, timeout: float,
) -> tuple[bytes, bytes, int]:
    """Wait for a spawned process, or kill it (and reap it) on timeout -- an unreaped
    kill leaves a zombie, and a leaked adb process holds the device connection open."""
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return b"", b"timed out", 124  # exit code 124, like timeout(1)
    return stdout, stderr, process.returncode or 0


def _check_exit(command: str, stderr: bytes, exit_code: int) -> None:
    """Shared raise for run_binary's non-zero exit -- the only piece of
    AdbTransport/CliDevice's run_binary bodies that isn't already `_communicate_or_kill`."""
    if exit_code != 0:
        raise RuntimeError(f"{command!r} failed (exit {exit_code}): {stderr.decode(errors='replace')[:200]}")


async def _finish(process: asyncio.subprocess.Process, timeout: float) -> RunResult:
    """Shared tail of AdbTransport.run / LocalShellTransport.run -- they differ only in
    how the process is spawned."""
    stdout, stderr, exit_code = await _communicate_or_kill(process, timeout)
    return RunResult(
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
        exit_code=exit_code,
    )


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
        """A resident uiautomator2 companion server answers in ~0.2-0.3s; a fresh
        `uiautomator dump` process cold-starts in ~2.2s every call because it
        reloads the ART runtime each time."""
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
        return await _finish(process, timeout)

    async def run_binary(self, command: str, timeout: float = 15.0) -> bytes:
        """`exec-out` streams raw stdout bytes -- for commands like `screencap -p`
        whose output would be corrupted by `run`'s text decoding."""
        process = await asyncio.create_subprocess_exec(
            self._adb, "-s", self.serial, "exec-out", command,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr, exit_code = await _communicate_or_kill(process, timeout)
        _check_exit(command, stderr, exit_code)
        return stdout


class LocalShellTransport:
    async def describe(self) -> str:
        return platform.platform()  # stdlib, not a shell call

    async def run(self, command: str, timeout: float = 15.0) -> RunResult:
        process = await asyncio.create_subprocess_shell(
            command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        return await _finish(process, timeout)
