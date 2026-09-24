"""Goal scope, journal-dir resolution, and get_journal() wiring to core Journal."""

from __future__ import annotations

from pathlib import Path

import pytest
from typesymbolic.journal import Journal

from jevdevice.journal import decision_log
from jevdevice.journal.decision_log import goal_id_for, goal_scope


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    monkeypatch.setattr(decision_log, "_default_journal", None)
    monkeypatch.delenv(decision_log.ENV_JOURNAL_DIR, raising=False)
    monkeypatch.delenv(decision_log.ENV_MAX_AGE_DAYS, raising=False)
    monkeypatch.delenv(decision_log.ENV_ENABLED, raising=False)


def test_goal_scope_sets_stable_id_and_text() -> None:
    assert decision_log.current_goal() == (None, None)
    with goal_scope("open the calculator"):
        assert decision_log.current_goal() == (goal_id_for("open the calculator"), "open the calculator")
        with goal_scope("press 7"):  # nested scopes restore correctly
            assert decision_log.current_goal() == (goal_id_for("press 7"), "press 7")
        assert decision_log.current_goal() == (goal_id_for("open the calculator"), "open the calculator")
    assert decision_log.current_goal() == (None, None)


def test_goal_id_is_stable_and_content_derived() -> None:
    assert goal_id_for("press 7") == goal_id_for("press 7")
    assert goal_id_for("press 7") != goal_id_for("press 8")


def test_journal_dir_defaults_and_env_overrides(tmp_path: Path, monkeypatch) -> None:
    assert decision_log.journal_dir() == decision_log.DEFAULT_JOURNAL_DIR
    monkeypatch.setenv(decision_log.ENV_JOURNAL_DIR, str(tmp_path / "custom"))
    assert decision_log.journal_dir() == tmp_path / "custom"


def test_get_journal_returns_core_journal_writing_to_the_new_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(decision_log.ENV_JOURNAL_DIR, str(tmp_path))
    journal = decision_log.get_journal()
    assert isinstance(journal, Journal)
    assert journal.root == tmp_path
    assert journal is decision_log.get_journal()  # process-wide singleton

    journal.record_decision(call_id="c1", engine="jev", phase="gate", answers={})
    rows = list(journal.replay())
    assert rows and rows[0]["call_id"] == "c1"
    assert list(tmp_path.glob("journal-*.jsonl"))  # daily rotation


def test_get_journal_disabled_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(decision_log.ENV_JOURNAL_DIR, str(tmp_path))
    monkeypatch.setenv(decision_log.ENV_ENABLED, "0")
    journal = decision_log.get_journal()
    journal.record_decision(call_id="c1", engine="jev", phase="gate", answers={})
    assert list(tmp_path.glob("journal-*.jsonl")) == []
