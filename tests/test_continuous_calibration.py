"""Phase 4.5 tests: the continuous calibration loop's knobs, segment tags,
refit proposals, shadow-run comparison, and the MECHANICAL tighten-only
promotion asymmetry (Standing rule 3 as a test, not a convention). All tests
run against synthetic journal rows -- the loop itself never calls a model."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from jevdevice.budget import LAYA_PROFILE
from jevdevice.calibrate import continuous as cont
from jevdevice.calibrate.continuous import (
    DEV,
    HUMAN,
    LOOSEN,
    TIGHTEN,
    UNRESOLVED,
    PromotionError,
    SignalStats,
    apply_proposal,
    classify,
    conformal_spike,
    persist,
    promote,
    select_window,
    shadow_compare,
    tag_segments,
)
from jevdevice.journal.decision_log import CALIBRATION, DecisionJournal


def _decision_row(call_id: str, *, phase: str = "gate", state: dict | None = None,
                  answers: dict | None = None, engine: str = "laya",
                  ts: datetime | None = None) -> dict:    return {
        "type": "decision",
        "ts": _naive(ts or _NOW).isoformat(timespec="milliseconds"),  # journal ts are naive-local, same clock family as DecisionJournal writes
        "call_id": call_id,
        "goal_id": None,
        "goal": None,
        "engine": engine,
        "model_revision": "rev-a",
        "phase": phase,
        "state": state,
        "questions": None,
        "answers": answers,
        "truncation": None,
        "usage": None,
        "error": None,
    }


_NOW = datetime(2026, 9, 20, 12, 0, 0)  # noqa: DTZ001 -- journal ts are naive-local, same clock family as DecisionJournal writes


def _naive(value: datetime) -> datetime:
    """Journal timestamps are naive-local (DecisionJournal's default clock); the
    synthetic fixtures use the same clock family so window math is comparable."""
    return value


def _outcome_row(call_id: str, verification: str) -> dict:
    return {"type": "outcome", "ts": "2026-09-20T12:00:01.000", "call_id": call_id,
            "goal_id": None, "goal": None, "device": None, "executed_command": None,
            "verification": verification, "status": None, "recovery_command": None,
            "graph_edge": None, "decision": None, "reasons": None,
            "exit_code": None, "satisfied": None}


# ---- window knobs (T1) ------------------------------------------------------------


def test_window_is_both_a_day_and_a_row_knob() -> None:
    now = datetime(2026, 9, 20)  # noqa: DTZ001 -- same clock family as the journal's naive-local ts
    old = [f"ts-{i}" for i in range(0)]  # rows older than the window
    rows = [_decision_row(str(i), ts=now - timedelta(days=40)) for i in range(10)]
    rows += [_decision_row(str(100 + i), ts=now - timedelta(days=1)) for i in range(5)]
    window, provenance = select_window(rows, now=now, window_days=30)
    assert len(window) == 5 and old == []
    assert provenance["effective_span_days"] == 1
    assert provenance["window_days_requested"] == 30
    # a journal younger than the window is used whole
    young = [_decision_row(str(i), ts=now - timedelta(hours=i + 1)) for i in range(7)]
    window, provenance = select_window(young, now=now, window_days=30)
    assert len(window) == 7
    assert provenance["effective_span_days"] == 0


# ---- segment tags (T2) -------------------------------------------------------------


def test_segments_split_device_verified_from_human_resolved() -> None:
    rows = [
        _decision_row("a"), _outcome_row("a", "verified"),
        _decision_row("b"), _outcome_row("b", "escalated"),
        _decision_row("c"), _outcome_row("c", "failed"),
        _decision_row("d"), _outcome_row("d", "none"),
        _decision_row("e"),  # no outcome row at all
    ]
    segments = tag_segments(rows)
    assert segments["a"].kind == DEV
    assert segments["b"].kind == HUMAN
    assert segments["c"].kind == DEV          # device refuted it -- still device truth
    assert segments["d"].kind == UNRESOLVED   # ran, nothing confirmed it
    assert cont.segment_of(segments, "e") == UNRESOLVED  # no outcome row at all
    assert cont.segment_of({}, "f") == UNRESOLVED


def test_signal_stats_carry_their_row_segments() -> None:
    rows = [
        _decision_row("a", state={"proposed_command": "svc bluetooth disable"},
                      answers={"safe": {"type": "noul", "noul": 0.9}}),
        _outcome_row("a", "verified"),
        _decision_row("b", state={"proposed_command": "svc bluetooth disable"},
                      answers={"safe": {"type": "noul", "noul": 0.9}}),
        _outcome_row("b", "escalated"),
    ]
    stats = cont.build_signals(rows, tag_segments(rows))
    assert stats["gate noul"].segments == [DEV, HUMAN]


# ---- sweeps + proposals (T1) --------------------------------------------------------


def test_thin_sample_warns_and_keeps_collecting() -> None:
    stats = SignalStats("gate noul", values=[0.9] * 19, labels=[True] * 19)
    stats.sweep()
    assert stats.proposed_threshold is None
    assert "keep collecting" in stats.warn


def test_proposal_needs_the_precision_bar_not_just_accuracy() -> None:
    # 18/20 correct overall, but no threshold reaches 0.95 precision on
    # enough rows: the misses sit right among the hits at every cut.
    values = [0.95] * 17 + [0.9] + [0.1] * 2
    labels = [True] * 18 + [False] * 2
    stats = SignalStats("gate noul", values=values, labels=labels)
    stats.sweep()
    assert stats.proposed_threshold is None  # 17/18 = 0.944 < 0.95 at t=0.9
    assert "unreachable" in stats.warn


def test_sweep_picks_the_loosest_qualifying_threshold() -> None:
    values = [0.9] * 20 + [0.3] * 5
    labels = [True] * 20 + [False] * 5
    stats = SignalStats("gate noul", values=values, labels=labels)
    stats.sweep()
    assert stats.proposed_threshold == 0.35  # first t that excludes the 0.3 negatives
    assert stats.proposed_precision == 1.0


# ---- tighten-only promotion (T4, mechanical) ----------------------------------------


def test_every_calibration_knob_tightens_upward() -> None:
    assert classify("min_confidence", 0.6, 0.7) == TIGHTEN
    assert classify("min_confidence", 0.6, 0.5) == LOOSEN
    assert classify("gate_threshold", 0.8, 0.8) == "equal"


def test_loosening_proposal_is_refused_without_signoff() -> None:
    proposal = cont.Proposal(engine="laya", values={"min_confidence": 0.4},
                             evidence={}, provenance={"window_days_requested": 30})
    with pytest.raises(PromotionError, match="loosens"):
        apply_proposal(LAYA_PROFILE, proposal)


def test_mixed_proposal_is_refused_too() -> None:
    proposal = cont.Proposal(
        engine="laya", values={"min_confidence": 0.7, "min_margin": 0.05},
        evidence={}, provenance={"window_days_requested": 30})
    with pytest.raises(PromotionError):
        apply_proposal(LAYA_PROFILE, proposal)


def test_tightening_promotion_needs_no_signoff() -> None:
    proposal = cont.Proposal(engine="laya", values={"min_confidence": 0.7},
                             evidence={}, provenance={"window_days_requested": 30})
    promoted = promote(LAYA_PROFILE, proposal)
    assert promoted.min_confidence == 0.7
    assert promoted.gate_threshold == LAYA_PROFILE.gate_threshold  # untouched knobs stay
    assert LAYA_PROFILE.min_confidence == 0.6                      # incumbent immutable


def test_loosening_with_human_signoff_passes() -> None:
    proposal = cont.Proposal(engine="laya", values={"min_confidence": 0.4},
                             evidence={}, provenance={"window_days_requested": 30},
                             human_signoff=True)
    promoted = promote(LAYA_PROFILE, proposal)
    assert promoted.min_confidence == 0.4


def test_promotion_refuses_an_unprovenanced_proposal() -> None:
    proposal = cont.Proposal(engine="laya", values={"min_confidence": 0.7},
                             evidence={}, provenance={})
    with pytest.raises(PromotionError, match="provenance"):
        promote(LAYA_PROFILE, proposal)


def test_signoff_env_is_the_human_switch(monkeypatch) -> None:
    monkeypatch.delenv(cont.ENV_PROMOTION_SIGNOFF, raising=False)
    assert cont.signoff_from_env() is False
    monkeypatch.setenv(cont.ENV_PROMOTION_SIGNOFF, "signoff")
    assert cont.signoff_from_env() is True


# ---- shadow run (T3) ----------------------------------------------------------------


def test_shadow_compare_records_both_verdicts_and_flips() -> None:
    stats = {"gate noul": SignalStats(
        "gate noul",
        values=[0.95] * 10 + [0.75] * 5 + [0.3] * 10,
        labels=[True] * 10 + [True] * 5 + [False] * 10)}
    proposed = {"gate_threshold": 0.7}  # incumbent 0.8: the 0.75s would be approved
    shadow = shadow_compare(LAYA_PROFILE, proposed, stats)
    entry = shadow["gate_threshold"]
    assert entry["current_verdict"] == {"approved": 10, "precision": 1.0}
    assert entry["proposed_verdict"] == {"approved": 15, "precision": 1.0}
    assert entry["escalated_under_current_approved_under_proposed"] == 5
    assert entry["approved_under_current_escalated_under_proposed"] == 0


def test_persist_writes_a_record_and_a_calibration_row(tmp_path) -> None:
    provenance = {"generated_at": "2026-09-20T12:00:00", "engine": "laya", "n": 430}
    proposal = cont.Proposal(engine="laya", values={"min_confidence": 0.7},
                             evidence={}, provenance=provenance)
    path = persist(tmp_path, provenance, proposal)
    assert json.loads(path.read_text())["proposal_values"] == {"min_confidence": 0.7}
    rows = [r for r in DecisionJournal(tmp_path).replay() if r.get("type") == CALIBRATION]
    assert len(rows) == 1
    assert rows[0]["event"] == "proposal"
    assert rows[0]["result"]["shadow"] is None          # no evidence sample -> no shadow
    assert rows[0]["result"]["record"].endswith(path.name)


def test_persist_records_a_no_proposal_run(tmp_path) -> None:
    provenance = {"generated_at": "2026-09-20T12:00:00", "engine": "laya", "n": 1}
    path = persist(tmp_path, provenance, None)
    assert json.loads(path.read_text())["event"] == "no-proposal"
    rows = [r for r in DecisionJournal(tmp_path).replay() if r.get("type") == CALIBRATION]
    assert rows[0]["event"] == "no-proposal"
    assert "no-proposal" in path.name or json.loads(path.read_text())["event"] == "no-proposal"


# ---- conformal spike (T5) ------------------------------------------------------------


def test_spike_keeps_when_approval_mass_is_noise_thin() -> None:
    # perfect separation but only 8 approved rows: no adoptable guarantee
    stats = {"gate noul": SignalStats("gate noul",
                                      values=[0.9] * 8 + [0.1] * 12,
                                      labels=[True] * 8 + [False] * 12)}
    report = conformal_spike(stats)
    entry = report["signals"]["gate noul"]
    assert entry["verdict"] == "keep"
    assert any("approval mass" in reason for reason in entry["reasons"])


def test_spike_adopts_only_on_adequate_perfect_data() -> None:
    stats = {"gate noul": SignalStats("gate noul",
                                      values=[0.9] * 25 + [0.1] * 25,
                                      labels=[True] * 25 + [False] * 25)}
    report = conformal_spike(stats)
    entry = report["signals"]["gate noul"]
    assert entry["verdict"] == "adopt"
    assert report["verdict"] == "adopt"


def test_aci_tightens_on_bad_outcomes() -> None:
    # alternating labels: the first approval is bad -> alpha must move DOWN
    result = cont.aci_update([0.9, 0.9, 0.9, 0.9], [False, True, True, True])
    assert result["bad"] == 1
    assert result["alpha_final"] < cont.CONFORMAL_ALPHA


# ---- end-to-end loop on a synthetic journal -----------------------------------------


def test_run_emits_a_provenance_tagged_no_proposal(tmp_path, capsys) -> None:
    directory = tmp_path / "journal"
    journal = DecisionJournal(directory)
    # 18 clean hits at 0.9, one true-labeled row at 0.4, five false-labeled rows
    # at 0.8: no cut reaches 0.95 precision with >= 20 approved rows.
    plan = [("svc bluetooth disable", 0.9, "verified")] * 18 + [
        ("svc bluetooth disable", 0.4, "verified"),
    ] + [("svc data disable", 0.8, "escalated")] * 5
    for i, (command, noul, verification) in enumerate(plan):
        call_id = f"call-{i}"
        journal.record_decision(call_id=call_id, engine="laya", model_revision="rev-a",
                                phase="gate", state={"proposed_command": command},
                                answers={"safe": {"type": "noul", "noul": noul}})
        journal.record_outcome(call_id=call_id, verification=verification)
    proposal = cont.run(directory, now=datetime(2026, 9, 20, 13, 0, 0))  # noqa: DTZ001 -- same clock family as the journal's naive-local ts
    out = capsys.readouterr().out
    assert proposal is None  # nothing clears 0.95 on this window
    assert "engine=laya" in out
    assert "device-verified" in out  # segment mix reported (T2)
    rows = [r for r in DecisionJournal(directory).replay() if r.get("type") == CALIBRATION]
    assert rows and rows[0]["event"] == "no-proposal"
    assert rows[0]["provenance"]["model_revision"] == "rev-a"
    assert rows[0]["provenance"]["segments"] == {"device-verified": 19, "human-resolved": 5}
    assert rows[0]["provenance"]["conformal_spike"]["verdict"] == "keep"


def test_run_proposes_and_journals_shadow_verdicts(tmp_path, capsys) -> None:
    directory = tmp_path / "journal"
    journal = DecisionJournal(directory)
    # 25 true-labeled taps at 0.95, 5 false-labeled ones at 0.2: a clean cut at
    # 0.35 qualifies (loosest sweep value that drops the negatives).
    plan = [("input tap 166 1394", "[66,1294][266,1494]", 0.95, "verified")] * 25 + [
        ("input tap 166 1394", "[813,1957][1013,2157]", 0.2, "escalated")] * 5
    for i, (command, bounds, noul, verification) in enumerate(plan):
        call_id = f"call-{i}"
        journal.record_decision(call_id=call_id, engine="laya", model_revision="rev-a",
                                phase="gate",
                                state={"proposed_command": command, "target_bounds": bounds},
                                answers={"safe": {"type": "noul", "noul": noul}})
        journal.record_outcome(call_id=call_id, verification=verification)
    proposal = cont.run(directory, now=datetime(2026, 9, 20, 13, 0, 0))  # noqa: DTZ001 -- same clock family as the journal's naive-local ts
    out = capsys.readouterr().out
    assert proposal is not None
    assert proposal.values["gate_threshold"] == 0.25  # loosest qualifying: drops the 0.2 negatives
    assert proposal.provenance["segments"]["device-verified"] == 25
    assert "shadow[gate_threshold]" in out
    rows = [r for r in DecisionJournal(directory).replay() if r.get("type") == CALIBRATION]
    assert rows[-1]["event"] == "proposal"
    shadow = rows[-1]["result"]["shadow"]["gate_threshold"]
    assert shadow["current_verdict"] == {"approved": 25, "precision": 1.0}  # incumbent 0.8
    assert shadow["proposed_verdict"] == {"approved": 25, "precision": 1.0}


# ---- the loop is offline-safe: zero model calls --------------------------------------


def test_module_never_imports_an_engine() -> None:
    banned = {"jevdevice.jev", "jevdevice.laya_backend", "jevdevice.mcp_server",
              "jevdevice.transport", "jevdevice.common"}
    code = (
        "import sys, jevdevice.calibrate.continuous as c\n"
        f"banned = {banned!r}\n"
        "print([m for m in banned if m in sys.modules])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ---- drift guard: the eval harness and the package module stay in lockstep -----------


def _load_eval_harness():
    path = Path(__file__).resolve().parent.parent / "eval" / "phases" / "recalibrate_thresholds.py"
    if not path.exists():
        pytest.skip("eval/phase4_recalibrate.py not present")
    spec = importlib.util.spec_from_file_location("recalibrate_thresholds_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dev_tap_case_tables_match_the_eval_harness() -> None:
    harness = _load_eval_harness()
    assert cont.GATE_NOUL_CASES == harness.GATE_NOUL_CASES
    assert cont.TAP_NOUL_CASES == harness.TAP_NOUL_CASES
    assert cont.NARROW_PICK_CASES == harness.NARROW_PICK_CASES


def test_labelers_agree_with_the_eval_harness() -> None:
    harness = _load_eval_harness()
    rows = [
        _decision_row("g1", state={"chosen_action": "disable bluetooth",
                                   "proposed_command": "svc bluetooth disable"},
                      answers={"safe": {"type": "noul", "noul": 0.7}}),
        _decision_row("g2", phase="ground",
                      state={"goal": "open Gmail", "candidates": {"gm": "com.google.gm"}},
                      answers={"fit_0": {"type": "noul", "noul": 0.6},
                               "pick": {"type": "choice", "choice": "gm",
                                        "confidence": 0.8,
                                        "probabilities": {"gm": 0.8, "none_of_these": 0.2}}}),
        _decision_row("g3", phase="kind",
                      state={"goal": "What is the battery level?"},
                      answers={"kind": {"type": "choice", "choice": "dumpsys",
                                        "confidence": 0.9,
                                        "probabilities": {"dumpsys": 0.9, "tap": 0.1}}}),
    ]
    labeled_rows = [dict(r, phase=r["phase"]) for r in rows]
    assert harness.label_gate_rows(labeled_rows) == [
        pair for row in rows for pair in cont._labeled_gate_nouls(row)]
    assert harness.label_fit_rows(labeled_rows) == [
        pair for row in rows for pair in cont._labeled_fit_nouls(row)]
    harness_picks = harness.label_pick_rows(labeled_rows)
    mine = [cont._labeled_choice_pick(row) for row in rows]
    mine = [p for p in mine if p]
    assert [(p["phase"], p["correct"]) for p in harness_picks] == \
           [(p["phase"], p["correct"]) for p in mine]
