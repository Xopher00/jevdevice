"""Pure functions plus the deny/read-only fast paths, no network — gate.py's
pre-checks must run and settle a verdict before any Jev call, so passing jev=None
here doubles as proof of that ordering: if either path touched jev, this would crash.
"""

from __future__ import annotations

import shlex

from jevdevice.judge.gate import (
    CommandVariant,
    GateVerdict,
    gate_command,
    is_denied,
    is_read_only,
)


def test_is_read_only_recognizes_known_prefixes() -> None:
    assert is_read_only("dumpsys battery")
    assert is_read_only("pm list packages")
    assert not is_read_only("svc bluetooth disable")


def test_is_denied_recognizes_known_substrings() -> None:
    assert is_denied("pm uninstall com.android.bluetooth")
    assert is_denied("svc wifi disable")
    assert is_denied("svc wifi 0")
    assert not is_denied("svc bluetooth disable")


def test_swipe_is_never_denied_structurally() -> None:
    """The original bug: a bare "wipe" deny entry matched inside "input swipe ...".
    Fixed at the root now -- "swipe" is a single real argv token, never equal to
    the word "wipe", so no leading-space patch is needed to keep this safe."""
    assert not is_denied("input swipe 500 800 500 200 300")
    assert not is_denied(f"input swipe {500} {800} {500} {800} 800")  # long-press shape


def test_deny_words_still_catch_the_real_word() -> None:
    assert is_denied("recovery --wipe_data")
    assert is_denied("some factory-reset now")


def test_embedded_deny_substring_inside_a_real_typed_value_is_not_denied() -> None:
    """Realistic shape from ui.py's propose_type: a real on-screen value can
    contain "rm "/"su "/"dd " as ordinary words. shlex.quote wraps the value in
    its own token, so it never contributes an argv[0] the deny-list checks."""
    for value in ("please rm the old draft first", "ask su support for help", "please add the item"):
        command = f"input tap 500 900 && input text {shlex.quote(value)}"
        assert not is_denied(command), command
        assert not is_read_only(command)


def test_chained_command_denied_if_any_invocation_is_denied() -> None:
    assert is_denied("input tap 1 2 && pm uninstall com.evil")


def test_chained_command_not_read_only_unless_every_invocation_is() -> None:
    assert not is_read_only("cat /sdcard/notes.txt && rm -rf /sdcard")
    assert is_denied("cat /sdcard/notes.txt && rm -rf /sdcard")


async def test_denied_command_is_denied_without_ever_calling_jev() -> None:
    result = await gate_command(None, CommandVariant("rm -rf /sdcard", "unrelated"), chosen_label="check battery level")
    assert result.verdict == GateVerdict.DENY


async def test_read_only_command_is_approved_without_ever_calling_jev() -> None:
    result = await gate_command(None, CommandVariant("dumpsys battery", "read battery"), chosen_label="read the battery level")
    assert result.verdict == GateVerdict.ACT
    assert result.reason == "read_only"


def test_batched_clear_keyevent_classifies_the_same_as_chained_form() -> None:
    """propose_type's clear sequence: one `input keyevent` call with many keycodes,
    not 200 `&&`-chained invocations -- same real argv shape, cheaper to compute."""
    command = f"input tap 500 900 && input keyevent 123{' 67' * 200} && input text {shlex.quote('weather in helsinki')}"
    assert not is_denied(command)
    assert not is_read_only(command)
