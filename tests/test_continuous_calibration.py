"""calibrate/continuous.py + calibrate/units.py + calibrate/cases.py: the
core-journal-backed refit loop, the calibration-unit map, and the dev-case
labelers -- all offline, zero model calls.

Tighten-only promotion, pooling, and calibration-row shape are typesymbolic's
own contract, covered there: tests/test_calibrate.py
(test_recalibrate_refuses_a_loosening_without_signoff,
test_recalibrate_applies_a_loosening_with_signoff,
test_tighten_only_threshold_flags_pooling_across_model_revisions) and
tests/test_journal.py (test_record_calibration_row_shape,
test_two_qids_sharing_a_calib_group_pool_their_labels). This file tests only
what's still jevdevice's: the knob->unit map, the dev-case labelers, and the
reporting functions kept here.
"""

from __future__ import annotations

import subprocess
import sys

from typesymbolic.calibration_store import CalibrationStore
from typesymbolic.journal import Journal

from jevdevice.budget import LAYA_PROFILE
from jevdevice.calibrate import cases
from jevdevice.calibrate import continuous as cont
from jevdevice.calibrate.units import CALIBRATION_UNITS, label_case, threshold

# ---- dev-case labelers (single source, cases.py) -------------------------


def test_gate_label_matches_known_cases() -> None:
    assert cases.gate_label("disable bluetooth", "svc bluetooth disable") is True
    assert cases.gate_label("disable bluetooth", "svc data disable") is False
    assert cases.gate_label("disable bluetooth", "unknown command") is None


def test_tap_label_keys_on_command_and_bounds() -> None:
    assert cases.tap_label("input tap 166 1394", "[66,1294][266,1494]") is True
    assert cases.tap_label("input tap 166 1394", "[813,1957][1013,2157]") is False
    assert cases.tap_label("input tap 0 0", "[0,0][1,1]") is None


def test_narrow_correct_true_pick_and_abstain_absent() -> None:
    assert cases.narrow_correct("open Gmail", "gm") is True
    assert cases.narrow_correct("open Gmail", "messaging") is False
    assert cases.narrow_correct("book me a flight to Paris", None) is True
    assert cases.narrow_correct("book me a flight to Paris", "camera") is False
    assert cases.narrow_correct("an unrecognized goal", "x") is None


# ---- calibration-unit map --------------------------------------------------


def test_every_budget_knob_maps_to_a_calibration_unit() -> None:
    for knob in ("gate_threshold", "min_fit", "min_confidence", "min_margin"):
        assert knob in CALIBRATION_UNITS


async def test_threshold_defaults_to_the_profile_knob_with_no_labels(tmp_path) -> None:
    journal = Journal(root=tmp_path / "journal", background_writes=False)
    value = await threshold(
        LAYA_PROFILE, "gate_threshold", journal=journal,
        calibration_store=CalibrationStore(tmp_path / "calibration"),
    )
    assert value == LAYA_PROFILE.gate_threshold


# ---- reporting functions (kept: ECE/Brier/log-loss/conformal, no temperature) --


def test_report_unit_computes_probability_quality() -> None:
    pairs = [(0.9, True)] * 10 + [(0.1, False)] * 10
    metrics = cont.report_unit("gate|noul_p", pairs)
    assert metrics["n"] == 20
    assert metrics["accuracy@0.5"] == 1.0
    assert metrics["brier"] < 0.02


def test_report_unit_handles_no_labels() -> None:
    assert cont.report_unit("gate|noul_p", []) == {"n": 0}


def test_spike_keeps_when_approval_mass_is_noise_thin() -> None:
    values = [0.9] * 8 + [0.1] * 12
    labels = [True] * 8 + [False] * 12
    report = cont.conformal_spike(values, labels)
    assert report["verdict"] == "keep"
    assert any("approval mass" in reason for reason in report["reasons"])


def test_spike_adopts_only_on_adequate_perfect_data() -> None:
    values = [0.9] * 25 + [0.1] * 25
    labels = [True] * 25 + [False] * 25
    report = cont.conformal_spike(values, labels)
    assert report["verdict"] == "adopt"


def test_aci_tightens_on_bad_outcomes() -> None:
    result = cont.aci_update([0.9, 0.9, 0.9, 0.9], [False, True, True, True])
    assert result["bad"] == 1
    assert result["alpha_final"] < cont.CONFORMAL_ALPHA


def test_signoff_env_is_the_human_switch(monkeypatch) -> None:
    monkeypatch.delenv(cont.ENV_PROMOTION_SIGNOFF, raising=False)
    assert cont.signoff_from_env() is False
    monkeypatch.setenv(cont.ENV_PROMOTION_SIGNOFF, "signoff")
    assert cont.signoff_from_env() is True


# ---- end to end: a synthetic core journal produces a proposal -------------


def test_run_tightens_the_gate_threshold_on_a_synthetic_journal(tmp_path) -> None:
    directory = tmp_path / "journal"
    journal = Journal(root=directory, rotation="daily", background_writes=False)
    # incumbent (0.8) would wrongly approve the 0.85 negatives -- the loosest
    # threshold that excludes them (0.9) is a TIGHTEN, applied automatically.
    plan = [(0.95, True)] * 25 + [(0.85, False)] * 5
    for i, (value, correct) in enumerate(plan):
        call_id = f"call-{i}"
        label_case(journal, call_id=call_id, engine="laya", knob="gate_threshold", value=value, correct=correct)

    calibration_store = CalibrationStore(tmp_path / "calibration")
    results = cont.run(directory, engine="laya", human_signoff=False, calibration_store=calibration_store)
    result = results["gate_threshold"]
    assert result.applied
    assert result.threshold > LAYA_PROFILE.gate_threshold  # tightened, not loosened
    assert result.proposal.precision == 1.0


def test_run_keeps_incumbent_with_no_labels(tmp_path) -> None:
    calibration_store = CalibrationStore(tmp_path / "calibration")
    results = cont.run(tmp_path / "empty-journal", engine="laya", human_signoff=False, calibration_store=calibration_store)
    for knob, result in results.items():
        assert result.threshold == getattr(LAYA_PROFILE, knob)
        assert not result.applied


# ---- offline-safe: zero model calls ---------------------------------------


def test_temperature_vectors_collect_verified_choice_and_skip_failed(tmp_path) -> None:
    from typesymbolic.domain import Verdict
    from typesymbolic.question import Answer, QuestionRef

    journal = Journal(root=tmp_path / "journal", rotation="daily", background_writes=False)
    ref = QuestionRef(qid="q", group="pick", scale="confidence")
    good = Answer.from_choice("q", "a", {"a": 0.7, "b": 0.3})
    bad = Answer.from_choice("q", "a", {"a": 0.6, "b": 0.4})
    journal.record_decision(
        call_id="c1", engine="laya", phase=None, answers={"k": good}, questions={"k": ref})
    journal.record_verdict(call_id="c1", verdict=Verdict(status="verified", tests=("k",)))
    journal.record_decision(
        call_id="c2", engine="laya", phase=None, answers={"k": bad}, questions={"k": ref})
    journal.record_verdict(call_id="c2", verdict=Verdict(status="failed", tests=("k",)))

    vectors = cont.temperature_vectors(journal, engine="laya")
    assert vectors == [({"a": 0.7, "b": 0.3}, "a")]
    t = cont.fit_temperature(vectors)
    assert t is not None and 0.1 <= t <= 3.0


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
