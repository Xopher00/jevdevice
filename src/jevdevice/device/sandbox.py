"""The shell family inside a Vercel Sandbox microVM: CliDevice over a remote
transport, so a wrong pick can only damage a disposable VM.

Needs the `sandbox` extra (`vercel-sandbox`) and VERCEL_TOKEN / VERCEL_TEAM_ID /
VERCEL_PROJECT_ID in the workspace .env. Hardware and host-settings goals get
the VM's answers, not this machine's -- run filesystem/process/text goals here.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta

from jevdevice.transport import RunResult

from .cli import CliDevice
from .protocol import DEFAULT_COMMAND_TIMEOUT_S

SANDBOX_DEVICE_NAME = "cli:sandbox"
SANDBOX_TIME_LIMIT = timedelta(minutes=30)
# Seconds past the in-VM timeout(1) before the SDK itself kills the process.
SANDBOX_KILL_GRACE_S = 5.0
TIMEOUT_EXIT_CODE = 124


class SandboxShellTransport:
    """LocalShellTransport's run() contract, executed in a sandbox. Each command
    runs under timeout(1) in the VM so a timeout reads exactly like the local
    one: exit 124, "timed out"."""

    def __init__(self, sandbox) -> None:
        self._sandbox = sandbox

    async def describe(self) -> str:
        return f"vercel-sandbox:{self._sandbox.name}"

    async def run(self, command: str, timeout: float = DEFAULT_COMMAND_TIMEOUT_S) -> RunResult:
        done = await self._sandbox.run_process(
            "timeout", ["--kill-after=2", str(timeout), "bash", "-lc", command],
            capture_output=True, kill_after=timeout + SANDBOX_KILL_GRACE_S,
        )
        if done.returncode == TIMEOUT_EXIT_CODE:
            return RunResult(stdout="", stderr="timed out", exit_code=TIMEOUT_EXIT_CODE)
        return RunResult(stdout=done.stdout or "", stderr=done.stderr or "", exit_code=done.returncode)


class SandboxDevice(CliDevice):
    """CliDevice with a sandbox transport. Named apart from every host so the
    journal never pools sandbox outcomes with this machine's."""

    def __init__(self, transport: SandboxShellTransport) -> None:
        super().__init__(transport)
        self.name = SANDBOX_DEVICE_NAME

    async def run_binary(self, command: str, timeout: float = DEFAULT_COMMAND_TIMEOUT_S) -> bytes:
        # run_process captures text only, so binary stdout crosses the wire as base64.
        result = await self._transport.run(f"set -o pipefail; ({command}) | base64 -w0", timeout)
        if result.exit_code != 0:
            raise RuntimeError(f"{command!r} failed (exit {result.exit_code}): {result.stderr[:200]}")
        return base64.b64decode(result.stdout)

    async def window_size(self) -> tuple[int, int]:
        return 80, 24


@asynccontextmanager
async def open_sandbox_device(*, snapshot_id: str | None = None) -> AsyncIterator[SandboxDevice]:
    """A fresh sandbox (or a copy of `snapshot_id`) with all network egress
    denied, destroyed on exit whatever happens inside."""
    from vercel.sandbox import NetworkPolicy, SandboxClient, SnapshotSource

    client = SandboxClient.create()
    try:
        sandbox = await client.create_sandbox(
            source=SnapshotSource(snapshot_id=snapshot_id) if snapshot_id else None,
            network_policy=NetworkPolicy.deny_all(),
            execution_time_limit=SANDBOX_TIME_LIMIT,
        )
        try:
            yield SandboxDevice(SandboxShellTransport(sandbox))
        finally:
            await sandbox.destroy()
    finally:
        await client.aclose()
