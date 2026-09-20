"""Small utilities every other module depends on: connecting to the real Jev API + device, and
applying the fixed no-action-below-threshold confidence policy to one Jev Choice pick.
"""

from __future__ import annotations

import os
from pathlib import Path

from .budget import (  # engine names live in budget.py
    ENGINE_ENV,
    ENGINES,
    JEV_ENGINE_NAME,
)
from .jev import JevClient
from .laya_backend import LayaClient
from .matching import confidence_gate
from .transport import AdbTransport


# The project's key file lives at the workspace root (typesafe/.env, one level
# above this repo) alongside ANDROID_SERIAL. Sourcing it here -- not relying on
# the shell having exported it -- is what makes one env file work everywhere:
# the MCP server, the CLI, and any script. Values already in the environment win.
def load_env_file(start: Path | None = None) -> Path | None:
    """Parse the nearest .env at/above `start` (default: this file's directory),
    setting KEY=VALUE pairs that aren't already in the environment. Blank lines
    and # comments are skipped; values are taken verbatim. Returns the file that
    was loaded, or None if there is none."""
    directory = start or Path(__file__).resolve().parent
    for candidate in (directory, *directory.parents):
        path = candidate / ".env"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
        return path
    return None

# Wireless adb's IP:port changes with the network (DHCP) -- ANDROID_SERIAL is adb's own
# standard env var for this, so `adb`/`adb -s` and this project always agree on the target device.
# No fallback default: a wrong guess would silently target someone else's device.
SERIAL = os.environ.get("ANDROID_SERIAL")

# Judge-engine selector: jev = the hosted API (needs TYPESAFE_AI_API), laya = the
# pinned in-process checkpoint (no key). Both clients are duck-type twins
# (ask/usage/aclose), no ABC. The names + the per-engine budget profiles live
# in budget.py.

# Compute device for the in-process engine. CPU is the default: this machine's
# 4 GiB card OOMs while loading the checkpoint, and a wrong guess must never
# silently fall back to a different device mid-run. JEV_DEVICE=cuda opts back in.
DEVICE_ENV = "JEV_DEVICE"
DEFAULT_DEVICE = "cpu"


def bootstrap(serial: str | None = SERIAL) -> tuple[JevClient | LayaClient, AdbTransport]:
    """Judge engine + device transport. The workspace .env is loaded first (env
    vars already set win), so TYPESAFE_AI_API and ANDROID_SERIAL work from any
    entry point, not only a shell that sourced the file. TYPESAFE_AI_API is
    required only for the jev engine -- the check lives in its branch, so
    JEV_ENGINE=laya boots with no key set (that is what lets the MCP server run
    fully on-box)."""
    load_env_file()
    serial = serial if serial is not None else os.environ.get("ANDROID_SERIAL")
    engine = (os.environ.get(ENGINE_ENV) or JEV_ENGINE_NAME).strip().lower()
    if engine not in ENGINES:
        raise SystemExit(f"{ENGINE_ENV} must be one of {', '.join(ENGINES)}, got {engine!r}")
    if engine == JEV_ENGINE_NAME:
        api_key = os.environ.get("TYPESAFE_AI_API")
        if not api_key:
            raise SystemExit("TYPESAFE_AI_API not set (required for the jev engine; JEV_ENGINE=laya runs in-process)")
        client: JevClient | LayaClient = JevClient(api_key)
    else:
        client = LayaClient(device=os.environ.get(DEVICE_ENV, DEFAULT_DEVICE))
    if not serial:
        raise SystemExit("ANDROID_SERIAL not set -- run `adb devices` and export ANDROID_SERIAL=<serial>")
    return client, AdbTransport(serial)


def gated(pick, escalate_prefix: str = "=== ESCALATED ===") -> str | None:
    """Apply the fixed no-action-below-threshold policy to one Jev Choice pick."""
    ok, reason = confidence_gate(pick.probabilities, pick.confidence)
    if not ok:
        print(f"{escalate_prefix} {reason}")
        return None
    return pick.choice
