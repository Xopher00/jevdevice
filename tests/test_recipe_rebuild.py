"""rebuild(): held-out/exclusion filtering, the checked-kind pre-check,
replace-not-merge semantics with a .bak, and dry-run. All offline, a
RecordingJournal duck-typing `replay()` like test_recipes_planner.py's.
"""

from __future__ import annotations

import json

import pytest

from jevdevice.execution.recipes import (
    ENV_RECIPES_DIR,
    RebuildReport,
    RecipeStore,
    rebuild,
)
from jevdevice.journal.decision_log import goal_id_for


@pytest.fixture(autouse=True)
def _isolated_recipes_dir(tmp_path, monkeypatch):
    # Safety net for the one RecipeStore(directory=None) case below.
    monkeypatch.setenv(ENV_RECIPES_DIR, str(tmp_path / "recipes"))


class RecordingJournal:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def replay(self):
        yield from self._rows


def outcome_row(goal: str, *, kind: str | None, verification: str, command: str | None = None,
                edge: dict | None = None, call_id: str | None = None, ts: str = "",
                satisfied: float | None = None) -> tuple[dict, dict | None]:
    """Mirrors tests/test_recipes_planner.py's helper, plus `satisfied` since
    that's what the pre-check filter reads (flattened `extra`, like the real journal)."""
    outcome = {
        "type": "outcome", "ts": ts, "call_id": call_id, "goal_id": goal_id_for(goal),
        "goal": goal, "device": "emulator", "executed_command": command,
        "status": None, "recovery_command": None,
        "graph_edge": edge, "decision": None, "key": "pick", "kind": kind, "tier": None,
        "recipe_id": None, "satisfied": satisfied,
    }
    verdict = {"type": "verdict", "call_id": call_id, "status": verification} if call_id else None
    return outcome, verdict


def journal_rows(*pairs: tuple[dict, dict | None]) -> list[dict]:
    flat: list[dict] = []
    for outcome, verdict in pairs:
        flat.append(outcome)
        if verdict is not None:
            flat.append(verdict)
    return flat


def tap_goal_rows(goal: str, call_id: str) -> list[dict]:
    return journal_rows(outcome_row(
        goal, kind="tap", verification="verified", command="input tap 5 700",
        edge={"from_node": "com.calc", "to_node": "com.calc"}, call_id=call_id, ts="t",
    ))


# --- held-out / exclusion filtering --------------------------------------------

def test_heldout_rows_are_filtered_not_raised(tmp_path) -> None:
    goal = "heldout goal"
    rows = tap_goal_rows(goal, "c1")
    store = RecipeStore(tmp_path)
    report = rebuild(store, RecordingJournal(rows), heldout_goal_ids=frozenset({goal_id_for(goal)}), dry_run=True)
    assert report.n_recipes == 0
    assert report.skipped_heldout == 1
    assert report.skipped_excluded == 0


def test_excluded_goal_ids_are_skipped(tmp_path) -> None:
    kept, excluded = "keep this goal", "exclude this goal"
    rows = tap_goal_rows(kept, "c1") + tap_goal_rows(excluded, "c2")
    store = RecipeStore(tmp_path)
    report = rebuild(store, RecordingJournal(rows), exclude=frozenset({goal_id_for(excluded)}), dry_run=True)
    assert report.n_recipes == 1
    assert report.skipped_excluded == 1


# --- checked-kind pre-check (keyevent/swipe/set_dnd need a real `satisfied`) ----

def test_keyevent_step_without_satisfied_is_dropped(tmp_path) -> None:
    goal = "go back"
    rows = journal_rows(outcome_row(
        goal, kind="keyevent", verification="verified", command="input keyevent KEYCODE_BACK",
        call_id="c1", ts="t", satisfied=None,
    ))
    report = rebuild(RecipeStore(tmp_path), RecordingJournal(rows), dry_run=True)
    assert report.n_recipes == 0


def test_keyevent_step_with_satisfied_is_kept(tmp_path) -> None:
    goal = "go back"
    rows = journal_rows(outcome_row(
        goal, kind="keyevent", verification="verified", command="input keyevent KEYCODE_BACK",
        call_id="c1", ts="t", satisfied=0.9,
    ))
    report = rebuild(RecipeStore(tmp_path), RecordingJournal(rows), dry_run=True)
    assert report.n_recipes == 1


def test_approved_tap_row_becomes_a_recipe() -> None:
    """tap isn't a checked kind: kind + executed_command + a verified verdict
    is enough, with no `satisfied` field at all."""
    goal = "tap the 7 button"
    rows = tap_goal_rows(goal, "c1")
    store = RecipeStore(directory=None)  # exercises the ENV_RECIPES_DIR path this fixture sets
    report = rebuild(store, RecordingJournal(rows), dry_run=True)
    assert report.n_recipes == 1


# --- replace semantics, backup, save -------------------------------------------

def test_rebuild_replaces_not_merges(tmp_path) -> None:
    store = RecipeStore(tmp_path)
    old_goal = "old stale recipe"
    rebuild(store, RecordingJournal(tap_goal_rows(old_goal, "cold")))  # populates the store once
    assert store.get(goal_id_for(old_goal)) is not None

    new_goal = "fresh recipe"
    report = rebuild(store, RecordingJournal(tap_goal_rows(new_goal, "cnew")))
    assert store.get(goal_id_for(old_goal)) is None  # replaced, not merged
    assert store.get(goal_id_for(new_goal)) is not None
    assert report.n_recipes == 1
    assert report.n_new == 1  # wasn't in the store before this rebuild


def test_rebuild_writes_backup_before_overwriting(tmp_path) -> None:
    store = RecipeStore(tmp_path)
    rebuild(store, RecordingJournal(tap_goal_rows("first goal", "c1")))
    assert not (tmp_path / "recipes.json.bak").exists()  # nothing to back up yet

    rebuild(store, RecordingJournal(tap_goal_rows("second goal", "c2")))
    backup = json.loads((tmp_path / "recipes.json.bak").read_text())
    assert goal_id_for("first goal") in backup
    current = json.loads((tmp_path / "recipes.json").read_text())
    assert goal_id_for("second goal") in current
    assert goal_id_for("first goal") not in current


# --- dry-run ---------------------------------------------------------------------

def test_dry_run_writes_nothing(tmp_path) -> None:
    store = RecipeStore(tmp_path)
    rebuild(store, RecordingJournal(tap_goal_rows("seed goal", "c1")))  # a real file to protect
    before = (tmp_path / "recipes.json").read_text()

    report = rebuild(store, RecordingJournal(tap_goal_rows("other goal", "c2")), dry_run=True)
    assert report.n_recipes == 1
    assert (tmp_path / "recipes.json").read_text() == before  # untouched
    assert not (tmp_path / "recipes.json.bak").exists()
    assert store.get(goal_id_for("seed goal")) is not None  # in-memory store also untouched


# --- report counts ---------------------------------------------------------------

def test_report_counts_are_consistent(tmp_path) -> None:
    heldout_goal, excluded_goal, kept_goal = "heldout goal x", "excluded goal x", "kept goal x"
    rows = (
        tap_goal_rows(heldout_goal, "ch")
        + tap_goal_rows(excluded_goal, "ce")
        + tap_goal_rows(kept_goal, "ck")
    )
    store = RecipeStore(tmp_path)
    report = rebuild(
        store, RecordingJournal(rows),
        heldout_goal_ids=frozenset({goal_id_for(heldout_goal)}),
        exclude=frozenset({goal_id_for(excluded_goal)}),
    )
    assert report == RebuildReport(
        n_recipes=1, n_new=1, skipped_heldout=1, skipped_excluded=1, dropped_unverified=0,
    )
