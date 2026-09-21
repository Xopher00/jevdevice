"""Phase 7 flywheel: the perturbation plan (T1), failure mining (T2), and the
training-format exporters (T3) — all tested offline on synthetic journals,
zero model calls, zero device. The held-out guard and shadow exclusion are
asserted where the real tooling enforces them."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys_path = str(REPO / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from jevdevice.decision_log import DecisionJournal


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p7h = _load("phase7_harness", REPO / "eval" / "phase7_harness.py")
p7m = _load("phase7_mine_failures", REPO / "eval" / "phase7_mine_failures.py")
p7e = _load("phase7_export_flywheel", REPO / "eval" / "phase7_export_flywheel.py")


# --- T1: perturbation plan -----------------------------------------------------

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


# --- T1: trajectory extraction (report core) -----------------------------------

def _journal_with_run(tmp_path: Path, goal: str, *, verified: bool) -> DecisionJournal:
    journal = DecisionJournal(tmp_path)
    journal.record_decision(call_id="c1", engine="jev", model_revision="m", phase="kind",
                            state={"goal": goal}, questions={}, answers={},
                            goal=goal, goal_id=DecisionJournal.goal_id_for(goal)
                            if hasattr(DecisionJournal, "goal_id_for") else None)
    journal.record_outcome(call_id="c1", executed_command="svc bluetooth enable",
                           verification="verified" if verified else "escalated",
                           goal=goal)
    return journal


def test_extract_trajectories_windows_rows_and_scores_verification(tmp_path: Path) -> None:
    from jevdevice.decision_log import goal_id_for

    goal = "Turn Bluetooth on."
    journal = _journal_with_run(tmp_path, goal, verified=True)
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


# --- T2: failure mining --------------------------------------------------------

def _mine_journal(tmp_path: Path, goal: str) -> tuple[list[dict], dict]:
    from jevdevice.decision_log import goal_id_for

    journal = DecisionJournal(tmp_path)
    goal_id = goal_id_for(goal)
    journal.record_decision(call_id="fail-decision", engine="jev", model_revision="m", phase="kind",
                            state={"goal": goal}, questions={}, answers={}, goal=goal, goal_id=goal_id)
    journal.record_outcome(call_id="fail-decision", verification="escalated", status="escalated",
                           goal=goal, goal_id=goal_id)
    journal.record_decision(call_id="retry-decision", engine="jev", model_revision="m", phase="fill",
                            state={"goal": goal}, questions={}, answers={}, goal=goal, goal_id=goal_id)
    journal.record_outcome(call_id="retry-decision", executed_command="svc bluetooth enable",
                           verification="verified", goal=goal, goal_id=goal_id)
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
    journal = DecisionJournal(tmp_path)
    # escalation with NO later verified outcome for the same goal -> no pair
    journal.record_outcome(call_id="only-failure", verification="escalated",
                           goal="Open the Camera app.", goal_id="g-camera")
    # a held-out goal text is hard-skipped, never exported
    heldout = next(iter(p7m.heldout_goals()))
    journal.record_outcome(call_id="h", verification="escalated", goal=heldout.title(), goal_id="h1")
    pairs, dropped = p7m.collect(journal)
    assert pairs == []
    assert dropped["heldout_goal"] == 1


# --- T3: exporters -------------------------------------------------------------

def _rows_for_export(tmp_path: Path, goal: str) -> dict[str, list[dict]]:
    """A minimal journaled trajectory: kind decision -> outcome -> recovery."""
    def decision_row(call_id, phase, ts, choice=None):
        return {"type": "decision", "ts": ts, "call_id": call_id, "goal_id": "gid", "goal": goal,
                "phase": phase, "engine": "jev", "shadow_of": None,
                "answers": ({"kind": {"type": "choice", "choice": "toggle_service", "confidence": 0.9},
                             "any_fit": {"type": "noul", "noul": 0.8}}
                            if choice else {"satisfied": {"type": "noul", "noul": 0.7}}),
                "reasons": [] if choice else ["verify noul low"]}
    return {
        "gid": [
            decision_row("d1", "kind", "2026-09-21T02:00:00", choice=True),
            {"type": "outcome", "ts": "2026-09-21T02:00:01", "call_id": "d1", "goal_id": "gid",
             "goal": goal, "verification": "escalated", "status": "escalated", "reasons": ["gate"]},
            decision_row("d2", "fill", "2026-09-21T02:00:02", choice=True),
            {"type": "outcome", "ts": "2026-09-21T02:00:03", "call_id": "d2", "goal_id": "gid",
             "goal": goal, "verification": "verified", "executed_command": "svc bluetooth enable"},
        ]
    }


def test_span_weighted_exporter_weights_actions_over_chatter(tmp_path: Path) -> None:
    rows = _rows_for_export(tmp_path, "Turn Bluetooth on.")["gid"]
    row = p7e.span_weighted_row("gid", rows, "Turn Bluetooth on.")
    assert row is not None and row["format"] == "span-weighted-v1"
    roles = [s["role"] for s in row["spans"]]
    assert "action" in roles and "decision" in roles and "chatter" in roles
    weights = {s["role"]: s["weight"] for s in row["spans"]}
    assert weights["action"] == p7e.WEIGHT_ACTION == 1.0
    assert weights["chatter"] < weights["action"]  # chatter is discounted, never dropped
    # an unverified trajectory exports nothing
    unverified = [r for r in rows if r.get("verification") != "verified" or r.get("type") == "decision"]
    assert p7e.span_weighted_row("gid2", unverified, "g") is None


def test_sgcd_row_puts_loss_on_the_recovery_side_only(tmp_path: Path) -> None:
    goal = "Turn Bluetooth off."
    rows = _rows_for_export(tmp_path, goal)["gid"]
    pair = {
        "goal_id": "gid", "goal": goal, "provenance": "retry",
        "broken": {"ts": "2026-09-21T02:00:01", "call_id": "d1", "status": "escalated",
                   "reasons": ["gate"], "attempted_command": None,
                   "failed_trajectory_call_ids": ["d1"]},
        "recovery": {"ts": "2026-09-21T02:00:03", "call_id": "d2",
                     "executed_command": "svc bluetooth disable", "recovery_command": None,
                     "recovery_decision_call_ids": ["d2"], "device": "s"},
    }
    # sgcd_row re-reads the journal by goal_id; point it at the synthetic rows.
    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(p7e.p7m, "trajectories_by_goal", lambda journal: {"gid": rows})
        sgcd = p7e.sgcd_row(pair)
    assert sgcd["format"] == "sgcd-v1"
    # the broken prefix is context (kept, with the failure outcome's call_id),
    # and the target carries ONLY the recovery side's action/decision spans.
    assert sgcd["context"]["broken_state"]["call_id"] == "d1"
    assert {s["role"] for s in sgcd["target"]} == {"action", "decision"}
    assert all(s["call_id"] != "d1" for s in sgcd["target"])
    assert any(s["text"] == "svc bluetooth enable" for s in sgcd["target"])
    assert not any(s["role"] == "chatter" for s in sgcd["target"])
