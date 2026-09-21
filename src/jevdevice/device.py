"""The Device protocol: the seam between the judge engine and whatever it acts on.

Engine modules (dispatch, ui, elements, services, app_launch, outcomes, planner)
consume ONLY this protocol -- device knowledge lives in the implementations
(AdbDevice here, transport.py underneath it, which stays frozen and untouched).

Members are the audited, existing call sites -- nothing hypothetical (the second
implementation is the test of this interface):

- snapshot  -> `dump_hierarchy()`: the live UI tree XML (elements.dump_screen and
  every describe_screen/screen_summary consumer).
- execute   -> `run(command, timeout)`: one adb shell command. "Probe" is the
  same member run with a read-only command -- the gate classifies read-only
  commands, and no separate probe path exists to generalize.
- capture   -> `run_binary(command)`: raw stdout bytes (screencap -p).
- geometry  -> `window_size()`: swipe endpoints (ui.propose_swipe).
- name      -> the journal's per-device `device` column (outcomes.emit_outcome),
  so a second device family separates its rows without engine changes.

Two implementations exist: `AdbDevice` (the Android phone, over
AdbTransport) and `CliDevice` (the local shell, over LocalShellTransport). The
CLI one is the test of the interface's GUI-shaped seams, and it maps them
honestly rather than pretending:

- `dump_hierarchy()` returns a plain-TEXT snapshot (cwd + a listing), not a
  UI tree. The engine's XML parser rejects it and `pick_kind` degrades to an
  ungrounded kind pick (fail-closed) -- a CLI goal then abstains over the
  phone-closed ACTION_KINDS and escalates. A CLI goal's real execution path
  is the family-agnostic gated-command spine the engine already has
  (propose_from_closed_set -> gate_command -> device.run).
- `window_size()` is terminal geometry -- never consumed on CLI paths.
- `run_binary()` is raw stdout capture (the screencap analog).
- `run()` is the natural fit: exit code + stdout/stderr IS device truth.

Roadmap sketch names with NO current call site are deliberately absent and must
not be added until a real consumer exists: `describe()` (zero callers), `risk_class`
(never referenced), `fits()` (fit scores are judge Nouls over candidates, not
device truth), `settle()` (launch/verify backoff is call-site `delays` logic).
"""

from __future__ import annotations

import asyncio
import platform
import shutil
from typing import Protocol, runtime_checkable

from .transport import (
    AdbTransport,
    LocalShellTransport,
    RunResult,
    _communicate_or_kill,
)

# The command timeout every call site already relies on (transport.run's frozen
# default); named here so the protocol never hard-codes a bare number.
DEFAULT_COMMAND_TIMEOUT_S = 15.0


@runtime_checkable
class Device(Protocol):
    """What the engine needs from a device -- and nothing more."""

    name: str

    async def dump_hierarchy(self) -> str:
        """The live UI accessibility-tree XML (the snapshot)."""
        ...

    async def run(self, command: str, timeout: float = DEFAULT_COMMAND_TIMEOUT_S) -> RunResult:
        """One shell command (the execute/probe member); the exit code and
        stdout/stderr are the device's own truth about what happened."""
        ...

    async def run_binary(self, command: str, timeout: float = DEFAULT_COMMAND_TIMEOUT_S) -> bytes:
        """Raw stdout bytes for binary-output commands (screencap -p)."""
        ...

    async def window_size(self) -> tuple[int, int]:
        """(width, height) in pixels -- gesture geometry."""
        ...


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
        if exit_code != 0:
            raise RuntimeError(f"{command!r} failed (exit {exit_code}): {stderr.decode(errors='replace')[:200]}")
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
