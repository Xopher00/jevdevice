"""The Device seam, split by concern: `protocol.py` holds the interface the
engine types against, and one module per family holds its implementation
(`adb.py` -- the Android phone; `cli.py` -- the local shell; `sandbox.py` --
the shell family inside a Vercel Sandbox microVM).

Every family type is re-exported here, so engine modules, tests, and eval
tooling import them uniformly as `from jevdevice.device import ...`.

Adding a family means adding a module -- the engine is untouched.
"""

from __future__ import annotations

from .adb import AdbDevice
from .cli import CLI_SNAPSHOT_COMMAND, CliDevice
from .protocol import DEFAULT_COMMAND_TIMEOUT_S, Device
from .sandbox import (
    SANDBOX_DEVICE_NAME,
    SandboxDevice,
    SandboxShellTransport,
    open_sandbox_device,
)

__all__ = [
    "CLI_SNAPSHOT_COMMAND",
    "DEFAULT_COMMAND_TIMEOUT_S",
    "SANDBOX_DEVICE_NAME",
    "AdbDevice",
    "CliDevice",
    "Device",
    "SandboxDevice",
    "SandboxShellTransport",
    "open_sandbox_device",
]
