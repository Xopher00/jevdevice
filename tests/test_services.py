"""Pure functions only, no network, no device -- services.py's contract."""

from __future__ import annotations

from jevdevice.services import parse_key_value


def test_parse_key_value_reads_simple_one_colon_per_line() -> None:
    raw = "level: 64\nscale: 100\nhealth: 2"
    assert parse_key_value(raw) == {"level": "64", "scale": "100", "health": "2"}


def test_parse_key_value_extracts_real_dumpsys_settings_name_value_pairs() -> None:
    raw = "_id:2006 name:zen_mode pkg:android value:2 default:2 defaultSystemSet:true"
    assert parse_key_value(raw)["zen_mode"] == "2"


def test_parse_key_value_does_not_confuse_oldvalue_newvalue_with_a_bare_value_field() -> None:
    raw = "time: 09-19 16:03:40.934 mode:update oldValue:0:0:32 newValue:0:0:28 package:com.android.systemui"
    parsed = parse_key_value(raw)
    assert "zen_mode" not in parsed
    assert parsed["time"] == "09-19 16:03:40.934 mode:update oldValue:0:0:32 newValue:0:0:28 package:com.android.systemui"


def test_parse_key_value_handles_multiple_settings_entries_without_id_collisions() -> None:
    raw = (
        "_id:1 name:zen_mode pkg:android value:2\n"
        "_id:2 name:screen_brightness pkg:android value:120\n"
    )
    parsed = parse_key_value(raw)
    assert parsed["zen_mode"] == "2"
    assert parsed["screen_brightness"] == "120"
