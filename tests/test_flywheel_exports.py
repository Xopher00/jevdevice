"""The perturbation harness, failure mining, and training-format exporters --
all tested offline on synthetic journals, zero model calls, zero device. The
held-out guard and shadow exclusion are asserted where the real tooling
enforces them."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PHASES = REPO / "eval" / "phases"
if str(PHASES) not in sys.path:
    sys.path.insert(0, str(PHASES))

import export_flywheel as p7e
import mine_recoveries as p7m
import perturbation_harness as p7h
from typesymbolic.domain import ActOutcome, Verdict
from typesymbolic.journal import Journal
from typesymbolic.question import Answer

from jevdevice.journal.decision_log import goal_id_for

# --- perturbation plan ------------------------------------------------

def test_mutations_are_meaning_preserving_and_known_ids_fail_closed() -> None:
    assert p7h.apply_mutation("Open the Camera app.", "as_is") == "Open the Camera app."
    assert p7h.apply_mutation("Open the Camera app.", "lowercase") == "open the camera app."
    assert "Open the Camera app." in p7h.apply_mutation("Open the Camera app.", "polite")
    with pytest.raises(ValueError):
        p7h.apply_mutation("anything", "nonsense_id")


def test_plan_is_dev_only_with_perturbation_metadata() -> None:
    plan = p7h.build_plan()
    assert plan, "the plan must cover the dev phone goals"
    heldout_ids = {"pa02", "pa05", "pa07", "pa09", "pa10", "ph02", "ph04", "ph06", "ph08", "ph10", "ph12", "ph14"}
    source_ids = {run["source_goal_id"] for run in plan}
    assert not (source_ids & heldout_ids), "held-out goals must never enter the plan"
    # every run carries its perturbation metadata + a stable distinct goal_id
    for run in plan:
        assert run["mutation_id"] in p7h.WORDING_MUTATIONS
        assert len(run["goal_id"]) == 12
        assert run["run_id"].startswith(run["source_goal_id"] + "-")
    # each (goal, variant) is a distinct run; wording variants journal under
    # their own goal_id; same-text variants (baseline/start/pre_broken) share
    # the goal_id and are separated per-run by the report's ts window
    seen: set[tuple[str, str]] = set()
    for run in plan:
        key = (run["source_goal_id"], run["variant_id"])
        assert key not in seen, "duplicate run in the plan"
        seen.add(key)
    for run in plan:
        if run["mutation_id"] != "as_is":
            assert run["goal_id"] != p7h.__dict__["hashlib"].sha256(run["source_goal_text"].encode()).hexdigest()[:12], \
                "a wording mutation must journal under its own goal_id"


def test_plan_includes_pre_broken_and_start_state_variants() -> None:
    plan = p7h.build_plan()
    variant_ids = {run["variant_id"] for run in plan}
    assert "baseline" in variant_ids
    assert any(v.startswith("wording_") for v in variant_ids)
    assert "pre_broken" in variant_ids  # BT on/off and Camera-from-Calculator
    assert any(v.startswith("start_") for v in variant_ids)
    broken = {run["source_goal_id"]: run["setup_ids"] for run in plan if run["variant_id"] == "pre_broken"}
    assert broken["pa04"] == ["pa03"] and broken["pa03"] == ["pa04"]
    # a start-state setup may never BE the goal it sets up
    for run in plan:
        assert run["source_goal_id"] not in run["setup_ids"]


# --- real-journal plumbing (typesymbolic core rows, no wrapper) --------

def _journal(tmp_path: Path) -> Journal:
    """A tmp core Journal with a strictly-increasing clock (no ts ties)."""
    ticks = iter(range(1, 10_000))
    return Journal(root=tmp_path, background_writes=False,
                   clock=lambda: datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=next(ticks)))


def _decision(journal: Journal, call_id: str, goal: str, phase: str, *, choice: bool) -> None:
    """A primary decision row: kind pick (Choice) + fit chatter, or a bare verify Noul."""
    scope = {"goal": goal, "goal_id": goal_id_for(goal), "shadow_of": None}
    answers = ({"kind": Answer.from_choice("kind", "toggle_service", {"toggle_service": 0.9, "abstain": 0.1}),
                "any_fit": Answer.from_noul("any_fit", 0.8)}
               if choice else {"satisfied": Answer.from_noul("satisfied", 0.7)})
    journal.record_decision(call_id=call_id, engine="jev", phase=phase, scope=scope, answers=answers)


def _outcome(journal: Journal, call_id: str, goal: str, *, executed_command: str | None = None,
            status: str | None = None, act_reasons: tuple[str, ...] = ()) -> None:
    outcome = ActOutcome(succeeded=status == "ok", reasons=act_reasons)
    raw = {"goal": goal, "goal_id": goal_id_for(goal), "executed_command": executed_command, "status": status}
    extra = {k: v for k, v in raw.items() if v is not None}
    journal.record_outcome(call_id=call_id, gate=None, outcome=outcome, extra=extra or None)


# --- trajectory extraction (report core) ------------------------------------------

def test_extract_trajectories_windows_rows_and_scores_verification(tmp_path: Path) -> None:
    goal = "Turn Bluetooth on."
    journal = _journal(tmp_path)
    _decision(journal, "c1", goal, "kind", choice=True)
    _outcome(journal, "c1", goal, executed_command="svc bluetooth enable", status="ok")
    journal.record_verdict(call_id="c1", verdict=Verdict(status="verified"))
    runs = [{"run_id": "pa04-baseline", "goal_id": goal_id_for(goal), "goal": goal,
             "source_goal_id": "pa04", "variant_id": "baseline", "mutation_id": "as_is",
             "start_state": "home", "setup_ids": [], "status": "ok",
             "started_at": "0001-01-01T00:00:00", "finished_at": "9999-12-31T23:59:59"}]
    trajectories, status = p7h.extract_trajectories(journal, runs)
    assert len(trajectories) == 1 and trajectories[0]["rows"]
    assert status["pa04-baseline"]["device_verified"] is True
    # a row for a DIFFERENT goal (or outside the window) never joins the trajectory
    other = [{"run_id": "x", "goal_id": goal_id_for("unrelated"), "goal": "unrelated",
              "source_goal_id": "ph01", "variant_id": "baseline", "mutation_id": "as_is",
              "start_state": "home", "setup_ids": [], "status": "ok",
              "started_at": "0001-01-01T00:00:00", "finished_at": "9999-12-31T23:59:59"}]
    trajectories2, status2 = p7h.extract_trajectories(journal, other)
    assert trajectories2 == [] and status2["x"]["device_verified"] is False


# --- failure mining: failure mining --------------------------------------------------------

def _mine_journal(tmp_path: Path, goal: str) -> tuple[list[dict], dict]:
    journal = _journal(tmp_path)
    _decision(journal, "fail-decision", goal, "kind", choice=True)
    _outcome(journal, "fail-decision", goal, status="escalated")
    journal.record_verdict(call_id="fail-decision", verdict=Verdict(status="escalated"))
    _decision(journal, "retry-decision", goal, "fill", choice=True)
    _outcome(journal, "retry-decision", goal, executed_command="svc bluetooth enable", status="ok")
    journal.record_verdict(call_id="retry-decision", verdict=Verdict(status="verified"))
    return p7m.collect(journal)


def test_mining_pairs_a_failure_with_its_later_recovery(tmp_path: Path) -> None:
    goal = "Turn Bluetooth on."
    pairs, dropped = _mine_journal(tmp_path, goal)
    assert dropped.get("heldout_goal", 0) == 0
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["broken"]["call_id"] == "fail-decision"
    assert pair["broken"]["failed_trajectory_call_ids"] == ["fail-decision"]
    # the recovery references the failed trajectory by call_id chain
    assert pair["recovery"]["executed_command"] == "svc bluetooth enable"
    assert "fail-decision" not in pair["recovery"]["recovery_decision_call_ids"]
    assert pair["provenance"] == "retry"  # no recovery_command on this outcome


def test_mining_skips_unrecovered_failures_and_heldout(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    # escalation with NO later verified outcome for the same goal -> no pair
    _outcome(journal, "only-failure", "Open the Camera app.", status="escalated")
    journal.record_verdict(call_id="only-failure", verdict=Verdict(status="escalated"))
    # a held-out goal text is hard-skipped, never exported
    heldout = next(iter(p7m.heldout_goals()))
    _outcome(journal, "h", heldout.title(), status="escalated")
    journal.record_verdict(call_id="h", verdict=Verdict(status="escalated"))
    pairs, dropped = p7m.collect(journal)
    assert pairs == []
    assert dropped["heldout_goal"] == 1


# --- exporters: exporters -------------------------------------------------------------

def _rows_for_export(tmp_path: Path, goal: str) -> list[dict]:
    """A minimal journaled trajectory: kind decision -> outcome -> recovery."""
    journal = _journal(tmp_path)
    _decision(journal, "d1", goal, "kind", choice=True)
    _outcome(journal, "d1", goal, status="escalated", act_reasons=("gate",))
    _decision(journal, "d2", goal, "fill", choice=True)
    _outcome(journal, "d2", goal, executed_command="svc bluetooth enable", status="ok")
    return [r for r in journal.replay() if r.get("type") in ("decision", "outcome")]


def _row(rows: list[dict], call_id: str, row_type: str) -> dict:
    return next(r for r in rows if r.get("call_id") == call_id and r.get("type") == row_type)


def test_span_weighted_exporter_weights_actions_over_chatter(tmp_path: Path) -> None:
    rows = _rows_for_export(tmp_path, "Turn Bluetooth on.")
    verdicts = {"d2": "verified"}
    row = p7e.span_weighted_row("gid", rows, "Turn Bluetooth on.", verdicts)
    assert row is not None and row["format"] == "span-weighted-v1"
    roles = [s["role"] for s in row["spans"]]
    assert "action" in roles and "decision" in roles and "chatter" in roles
    weights = {s["role"]: s["weight"] for s in row["spans"]}
    assert weights["action"] == p7e.WEIGHT_ACTION == 1.0
    assert weights["chatter"] < weights["action"]  # chatter is discounted, never dropped
    # an unverified trajectory (its verified outcome dropped) exports nothing
    unverified = [r for r in rows if r.get("call_id") != "d2" or r.get("type") != "outcome"]
    assert p7e.span_weighted_row("gid2", unverified, "g", verdicts) is None


def test_sgcd_row_puts_loss_on_the_recovery_side_only(tmp_path: Path) -> None:
    goal = "Turn Bluetooth off."
    rows = _rows_for_export(tmp_path, goal)
    pair = {
        "goal_id": "gid", "goal": goal, "provenance": "retry",
        "broken": {"ts": _row(rows, "d1", "outcome")["ts"], "call_id": "d1", "status": "escalated",
                   "reasons": ["gate"], "attempted_command": None, "failed_trajectory_call_ids": ["d1"]},
        "recovery": {"ts": _row(rows, "d2", "outcome")["ts"], "call_id": "d2",
                     "executed_command": "svc bluetooth disable", "recovery_command": None,
                     "recovery_decision_call_ids": ["d2"], "device": "s"},
    }
    # point both the singleton journal and trajectories_by_goal at the synthetic
    # rows so no real ~/.jevdevice journal is ever touched.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(p7e.decision_log, "get_journal", lambda: _journal(tmp_path))
        monkeypatch.setattr(p7e.p7m, "trajectories_by_goal", lambda journal_rows: {"gid": rows})
        sgcd = p7e.sgcd_row(pair)
    assert sgcd["format"] == "sgcd-v1"
    # broken prefix is context; target carries ONLY the recovery side's spans.
    assert sgcd["context"]["broken_state"]["call_id"] == "d1"
    assert {s["role"] for s in sgcd["target"]} == {"action", "decision"}
    assert all(s["call_id"] != "d1" for s in sgcd["target"])
    assert any(s["text"] == "svc bluetooth enable" for s in sgcd["target"])
    assert not any(s["role"] == "chatter" for s in sgcd["target"])
