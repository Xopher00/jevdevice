"""The local-shell family: CliDevice over the frozen LocalShellTransport.

The honest member mapping (the point of this family -- it stresses the
protocol's GUI-shaped seams instead of flattering them):

- `dump_hierarchy()` returns a plain-TEXT snapshot (cwd + a listing), not a
  UI tree. The engine's XML parser rejects it and `pick_kind` degrades to an
  ungrounded kind pick (fail-closed) -- a CLI goal then abstains over the
  phone-closed ACTION_KINDS and escalates. A CLI goal's real execution path
  is the family-agnostic gated-command spine the engine already has
  (propose_from_closed_set -> gate_command -> device.run).
- `window_size()` is terminal geometry -- never consumed on CLI paths.
- `run_binary()` is raw stdout capture (the screencap analog).
- `run()` is the natural fit: exit code + stdout/stderr IS device truth.
"""

from __future__ import annotations

import asyncio
import platform
import shutil

from jevdevice.transport import (
    LocalShellTransport,
    RunResult,
    _check_exit,
    _communicate_or_kill,
)

from .protocol import DEFAULT_COMMAND_TIMEOUT_S

# The fixed read-only snapshot command a CLI device runs for its snapshot
# member -- the honest analog of a UI-tree dump (cwd + what's here). A fixed
# string the adapter itself owns: nothing caller-supplied ever rides it.
CLI_SNAPSHOT_COMMAND = "pwd && ls -A"


class CliDevice:
    """The local shell as a Device -- the second device family, wrapping the
    frozen LocalShellTransport (wrap, don't rewrite; it already had
    run/describe, and run_binary/window_size/dump_hierarchy live here because
    they are device-level, not transport-level, concerns).

    The name separates the journal's device column from every adb family:
    `cli:<hostname>`, stable across sessions."""

    def __init__(self, transport: LocalShellTransport | None = None) -> None:
        self._transport = transport if transport is not None else LocalShellTransport()
        self.name = f"cli:{platform.node()}"

    async def dump_hierarchy(self) -> str:
        """The plain-TEXT snapshot (the command + its real output), not a UI
        tree -- the engine's XML parse fails and pick_kind degrades to an
        ungrounded pick, fail-closed. See the module docstring for why this
        honest mismatch is the point of the CLI family."""
        result = await self._transport.run(CLI_SNAPSHOT_COMMAND)
        return f"$ {CLI_SNAPSHOT_COMMAND}\n{result.stdout}{result.stderr}"

    async def run(self, command: str, timeout: float = DEFAULT_COMMAND_TIMEOUT_S) -> RunResult:
        return await self._transport.run(command, timeout)

    async def run_binary(self, command: str, timeout: float = DEFAULT_COMMAND_TIMEOUT_S) -> bytes:
        """Raw stdout bytes -- the CLI analog of `screencap -p` streaming."""
        process = await self._spawn(command, timeout)
        stdout, stderr, exit_code = await _communicate_or_kill(process, timeout)
        _check_exit(command, stderr, exit_code)
        return stdout

    async def window_size(self) -> tuple[int, int]:
        """Terminal geometry -- the honest degenerate answer for a device with
        no screen. Never consumed on CLI paths (no swipes exist here)."""
        size = shutil.get_terminal_size(fallback=(80, 24))
        return size.columns, size.lines

    async def _spawn(self, command: str, timeout: float):
        return await asyncio.create_subprocess_shell(
            command, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
