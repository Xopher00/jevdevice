"""Small utilities every other module depends on: connecting to the real Jev API + device, and
applying the fixed no-action-below-threshold confidence policy to one Jev Choice pick.
"""

from __future__ import annotations

import os

from .jev import JevClient
from .matching import confidence_gate
from .transport import AdbTransport

# Wireless adb's IP:port changes with the network (DHCP) -- ANDROID_SERIAL is adb's own
# standard env var for this, so `adb`/`adb -s` and this project always agree on the target device.
# No fallback default: a wrong guess would silently target someone else's device.
SERIAL = os.environ.get("ANDROID_SERIAL")


def bootstrap(serial: str | None = SERIAL) -> tuple[JevClient, AdbTransport]:
    api_key = os.environ.get("TYPESAFE_AI_API")
    if not api_key:
        raise SystemExit("TYPESAFE_AI_API not set")
    if not serial:
        raise SystemExit("ANDROID_SERIAL not set -- run `adb devices` and export ANDROID_SERIAL=<serial>")
    return JevClient(api_key), AdbTransport(serial)


def gated(pick, escalate_prefix: str = "=== ESCALATED ===") -> str | None:
    """Apply the fixed no-action-below-threshold policy to one Jev Choice pick."""
    ok, reason = confidence_gate(pick.probabilities, pick.confidence)
    if not ok:
        print(f"{escalate_prefix} {reason}")
        return None
    return pick.choice
