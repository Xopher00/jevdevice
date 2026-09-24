"""Calibration units for jevdevice's budget knobs: the (group, scale) core
uses to pool labels, and the async threshold read through core's
`current_threshold` (opt-in `pool_revisions=True`, the accepted jevdevice
default) -- identical to the budget.py default until labels accumulate.
"""

from __future__ import annotations

from typesymbolic.calibrate import current_threshold
from typesymbolic.calibration_store import CalibrationStore
from typesymbolic.domain import Verdict
from typesymbolic.journal import Journal
from typesymbolic.question import Answer, QuestionRef, Scale

from jevdevice.budget import BudgetProfile
from jevdevice.journal import decision_log

# BudgetProfile field -> calibration unit. Every "safe"-family gate noul
# pools onto "gate" (H6): one gate_threshold knob per engine, not per qid.
CALIBRATION_UNITS: dict[str, tuple[str, Scale]] = {
    "gate_threshold": ("gate", "noul_p"),
    "min_fit": ("fit", "noul_p"),
    "min_confidence": ("pick", "confidence"),
    "min_margin": ("pick", "margin"),
}

_store: CalibrationStore | None = None


def store() -> CalibrationStore:
    global _store
    if _store is None:
        _store = CalibrationStore(decision_log.journal_dir().parent / "calibration")
    return _store


async def threshold(
    profile: BudgetProfile, knob: str, *, journal: Journal | None = None,
    calibration_store: CalibrationStore | None = None,
) -> float:
    """The calibrated value for one budget knob, or its profile default with
    no journal/labels yet -- gate outcomes unchanged until labels accumulate."""
    group, scale = CALIBRATION_UNITS[knob]
    return await current_threshold(
        journal=journal if journal is not None else decision_log.get_journal(),
        store=calibration_store or store(), group=group, scale=scale, engine=profile.engine,
        default_threshold=getattr(profile, knob), pool_revisions=True,
    )


def label_case(
    journal: Journal, *, call_id: str, engine: str, knob: str, value: float, correct: bool,
) -> None:
    """Tags an already-made call (a calibrate CLI's dev case) with the
    QuestionRef `current_threshold` needs to pool it, then verdicts it
    against the known ground truth -- a calibrate-CLI-only decision row,
    separate from the call's own (untagged) one."""
    group, scale = CALIBRATION_UNITS[knob]
    ref = QuestionRef(qid=f"calibrate.{knob}", group=group, scale=scale)
    if scale.startswith("noul"):
        answer = Answer(qid=ref.qid, type="noul", noul=value)
    elif scale == "margin":
        answer = Answer(qid=ref.qid, type="choice", confidence=value, probabilities={"a": value, "b": 0.0})
    else:
        answer = Answer(qid=ref.qid, type="choice", confidence=value, probabilities={"a": value})
    journal.record_decision(
        call_id=call_id, engine=engine, phase="calibrate-tag",
        answers={knob: answer}, questions={knob: ref},
    )
    journal.record_verdict(
        call_id=call_id, verdict=Verdict(status="verified" if correct else "failed", tests=(knob,)),
    )
