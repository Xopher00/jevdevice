"""The Device seam, split by concern: `protocol.py` holds the interface the
engine types against, and one module per family holds its implementation
(`adb.py` -- the Android phone; `cli.py` -- the local shell).

Everything the old single-file `device.py` exported is re-exported here, so
engine modules, tests, and eval tooling keep their existing
`from jevdevice.device import ...` imports unchanged.

Adding a family means adding a module -- the engine is untouched. A third
family (a Philips TV over JointSpace HTTP) is being prototyped under
`experiment/philips_tv/` (machine-local, gitignored) until its design is
audited through the same bar the second family cleared.
"""

from __future__ import annotations

from .adb import AdbDevice
from .cli import CLI_SNAPSHOT_COMMAND, CliDevice
from .protocol import DEFAULT_COMMAND_TIMEOUT_S, Device

__all__ = [
    "CLI_SNAPSHOT_COMMAND",
    "DEFAULT_COMMAND_TIMEOUT_S",
    "AdbDevice",
    "CliDevice",
    "Device",
]
