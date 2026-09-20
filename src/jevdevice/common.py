"""Small utilities every other module depends on: connecting to the real Jev API + device, and
applying the fixed no-action-below-threshold confidence policy to one Jev Choice pick.
"""

from __future__ import annotations

import os

from .jev import JevClient
from .laya_backend import LayaClient
from .matching import confidence_gate
from .transport import AdbTransport

# Wireless adb's IP:port changes with the network (DHCP) -- ANDROID_SERIAL is adb's own
# standard env var for this, so `adb`/`adb -s` and this project always agree on the target device.
# No fallback default: a wrong guess would silently target someone else's device.
SERIAL = os.environ.get("ANDROID_SERIAL")

# Judge-engine selector (Phase 2): jev = the Typesafe API (needs TYPESAFE_AI_API),
# laya = the pinned in-process checkpoint (no key). Default stays jev until P5's
# flip decision; both clients are duck-type twins (ask/usage/aclose), no ABC.
ENGINE_ENV = "JEV_ENGINE"
JEV_ENGINE_NAME = "jev"
LAYA_ENGINE_NAME = "laya"
ENGINES = (JEV_ENGINE_NAME, LAYA_ENGINE_NAME)


def bootstrap(serial: str | None = SERIAL) -> tuple[JevClient | LayaClient, AdbTransport]:
    """Judge engine + device transport. TYPESAFE_AI_API is required only for the
    jev engine -- the check lives in its branch, so JEV_ENGINE=laya boots with
    no key set (that is what lets the MCP server run fully on-box)."""
    engine = (os.environ.get(ENGINE_ENV) or JEV_ENGINE_NAME).strip().lower()
    if engine not in ENGINES:
        raise SystemExit(f"{ENGINE_ENV} must be one of {', '.join(ENGINES)}, got {engine!r}")
    if engine == JEV_ENGINE_NAME:
        api_key = os.environ.get("TYPESAFE_AI_API")
        if not api_key:
            raise SystemExit("TYPESAFE_AI_API not set (required for the jev engine; JEV_ENGINE=laya runs in-process)")
        client: JevClient | LayaClient = JevClient(api_key)
    else:
        client = LayaClient()
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
