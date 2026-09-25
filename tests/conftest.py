"""Isolates every test from the user's real ~/.jevdevice: JEV_JOURNAL_DIR
points at a per-test tmp dir, and the process-wide journal/calibration-store
singletons are reset so a prior test's cached instance is never reused."""

from __future__ import annotations

import pytest

from jevdevice.calibrate import units
from jevdevice.journal import decision_log


@pytest.fixture(autouse=True)
def _isolated_journal_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(decision_log.ENV_JOURNAL_DIR, str(tmp_path / "tsjournal"))
    monkeypatch.setattr(decision_log, "_default_journal", None)
    monkeypatch.setattr(units, "_store", None)
    yield
