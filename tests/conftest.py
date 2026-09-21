"""Test-suite-wide defaults, set before any test module imports the package.

JEV_SHADOW=0: the shadow observer attaches in common.bootstrap(), which
runs at import of jevdevice.mcp_server (pulled in by earlier test modules).
Tests must not load the laya checkpoint in the background of unrelated tests,
so the shadow defaults off here; tests/test_shadow.py controls the knob
explicitly per test via monkeypatch.
"""

import os

os.environ.setdefault("JEV_SHADOW", "0")
