"""Transport: the only thing hardwired per device type.

Everything else — what to probe, how to read the output, what command performs
an action, how to verify it — is generated per goal, not written in advance.
A new device type means one new Transport, nothing else.
"""

from __future__ import annotations

import asyncio
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

    async def describe(self) -> str:
        model = await self.run("getprop ro.product.model")
        release = await self.run("getprop ro.build.version.release")
        return f"Android {release.stdout.strip()}, {model.stdout.strip()}"

    async def run(self, command: str, timeout: float = 15.0) -> RunResult:
        process = await asyncio.create_subprocess_exec(
            self._adb, "-s", self.serial, "shell", command,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            return RunResult(stdout="", stderr="timed out", exit_code=124)
        return RunResult(stdout=stdout.decode(errors="replace"), stderr=stderr.decode(errors="replace"), exit_code=process.returncode or 0)


class LocalShellTransport:
    async def describe(self) -> str:
        uname = await self.run("uname -a")
        return uname.stdout.strip()

    async def run(self, command: str, timeout: float = 15.0) -> RunResult:
        process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            return RunResult(stdout="", stderr="timed out", exit_code=124)
        return RunResult(stdout=stdout.decode(errors="replace"), stderr=stderr.decode(errors="replace"), exit_code=process.returncode or 0)
