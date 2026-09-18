"""Command classification: read-only allowlist and a hard deny-list.

Both are applied to the literal command string regardless of what any tool or
model calls it — classification never trusts a self-reported label.
"""

from __future__ import annotations

# Fail-closed: an unrecognized command is treated as mutating, never read-only.
READ_ONLY_PREFIXES = (
    "pm list", "pm dump", "pm path", "pm resolve",
    "dumpsys", "getprop", "settings get", "cmd package resolve-activity",
    "uiautomator dump", "cat ", "ls ", "ls\t", "echo ", "wm size", "wm density",
    "input keyevent 3",
)

DENY_SUBSTRINGS = (
    "pm clear", "pm uninstall", "rm ", "rm\t", "reboot", "wipe", "factory",
    "settings put secure", "settings put global", " su ", "su -", "dd ",
    "mkfs", "format ", "> /dev/", "chmod 777",
    # Disabling wifi killed a live wireless-adb session mid-session (2026-09-18):
    # the recovery command couldn't reach a device whose network just died.
    "svc wifi disable", "svc wifi 0",
)


def is_read_only(command: str) -> bool:
    stripped = command.strip()
    return any(stripped.startswith(prefix) for prefix in READ_ONLY_PREFIXES)


def is_denied(command: str) -> bool:
    lowered = command.casefold()
    return any(bad in lowered for bad in DENY_SUBSTRINGS)
