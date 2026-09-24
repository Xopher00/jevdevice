"""The Device protocol: the seam between the judge engine and whatever it acts on.

Engine modules (dispatch, ui, elements, services, app_launch, outcomes, planner)
consume ONLY this protocol -- device knowledge lives in the family modules
(adb.py, cli.py), each wrapping the transport it owns.

Members are the audited, existing call sites -- nothing hypothetical (the second
implementation was the test of this interface):

- snapshot  -> `dump_hierarchy()`: the live UI tree XML (elements.dump_screen and
  every describe_screen/screen_summary consumer).
- execute   -> `run(command, timeout)`: one shell command. "Probe" is the same
  member run with a read-only command -- the gate classifies read-only
  commands, and no separate probe path exists to generalize.
- capture   -> `run_binary(command)`: raw stdout bytes (screencap -p).
- geometry  -> `window_size()`: swipe endpoints (ui.propose_swipe).
- name      -> the journal's per-device `device` column (outcomes.record_action),
  so a second device family separates its rows without engine changes.

Roadmap sketch names with NO current call site are deliberately absent and must
not be added until a real consumer exists: `describe()` (zero callers), `risk_class`
(never referenced), `fits()` (fit scores are judge Nouls over candidates, not
device truth), `settle()` (launch/verify backoff is call-site `delays` logic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # the protocol stays a pure seam: no runtime transport import
    from jevdevice.transport import RunResult

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
