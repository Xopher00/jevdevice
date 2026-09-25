"""LabelTarget: record_action(label=...) must label the pick's answer key(s)
and the gate's own key -- merged into one verdict when they share a call_id,
split into two verdicts when they don't. No `label` arg keeps the old,
single-key behaviour unchanged."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TYPESAFE_AI_API", "placeholder-for-import")
os.environ.setdefault("ANDROID_SERIAL", "placeholder-for-import")

from typesymbolic.journal import Journal
from typesymbolic.question import Answer, QuestionRef

from jevdevice.journal import decision_log, outcomes
from jevdevice.journal.outcomes import LabelTarget


def _journal(tmp_path: Path, monkeypatch) -> Journal:
    journal = Journal(root=tmp_path, background_writes=False)
    monkeypatch.setattr(decision_log, "_default_journal", journal)
    return journal


def _decision_row(call_id: str) -> dict:
    """One decision row with pick (choice), fit_1 (noul), safe_1 (noul)
    answer keys tagged into the pick/fit/gate calibration groups."""
    return {
        "call_id": call_id, "engine": "jev", "phase": "fill", "model_revision": None,
        "answers": {
            "pick": Answer.from_choice(qid="tap.pick", choice="a", probabilities={"a": 0.8}, confidence=0.8),
            "fit_1": Answer.from_noul(qid="tap.fit", noul=0.7),
            "safe_1": Answer.from_noul(qid="tap.safe_1", noul=0.6),
        },
        "questions": {
            "pick": QuestionRef(qid="tap.pick", group="pick", scale="confidence"),
            "fit_1": QuestionRef(qid="tap.fit", group="fit", scale="noul_p"),
            "safe_1": QuestionRef(qid="tap.safe_1", group="gate", scale="noul_p"),
        },
    }


async def test_label_target_fused_gate_labels_pick_fit_and_gate(tmp_path, monkeypatch) -> None:
    journal = _journal(tmp_path, monkeypatch)
    journal.record_decision(**_decision_row("cid-1"))

    label = LabelTarget(call_id="cid-1", keys=("pick", "fit_1"), gate_call_id="cid-1", gate_key="safe_1")
    outcomes.record_action(call_id="unused", key="unused", label=label, response={"status": "ok"})

    assert journal.labeled_pairs("pick", "confidence", engine="jev") == [(0.8, True)]
    assert journal.labeled_pairs("fit", "noul_p", engine="jev") == [(0.7, True)]
    assert journal.labeled_pairs("gate", "noul_p", engine="jev") == [(0.6, True)]


async def test_label_target_separate_gate_call_id_writes_two_verdicts(tmp_path, monkeypatch) -> None:
    journal = _journal(tmp_path, monkeypatch)
    journal.record_decision(**_decision_row("cid-pick"))
    journal.record_decision(**_decision_row("cid-gate"))

    label = LabelTarget(call_id="cid-pick", keys=("pick", "fit_1"), gate_call_id="cid-gate", gate_key="safe_1")
    outcomes.record_action(label=label, response={"status": "ok"})

    assert journal.labeled_pairs("pick", "confidence", engine="jev") == [(0.8, True)]
    assert journal.labeled_pairs("fit", "noul_p", engine="jev") == [(0.7, True)]
    # both decision rows carry a "gate" answer -- only cid-gate's verdict labels it
    assert journal.labeled_pairs("gate", "noul_p", engine="jev") == [(0.6, True)]
    verdict_rows = [row for row in journal.replay() if row["type"] == "verdict"]
    assert {tuple(row["tests"]) for row in verdict_rows} == {("pick", "fit_1"), ("safe_1",)}
    assert {row["call_id"] for row in verdict_rows} == {"cid-pick", "cid-gate"}


async def test_label_target_failed_response_labels_false(tmp_path, monkeypatch) -> None:
    journal = _journal(tmp_path, monkeypatch)
    journal.record_decision(**_decision_row("cid-fail"))

    label = LabelTarget(call_id="cid-fail", keys=("pick", "fit_1"), gate_call_id="cid-fail", gate_key="safe_1")
    outcomes.record_action(label=label, response={"status": "unverified", "exit_code": 1})

    assert journal.labeled_pairs("pick", "confidence", engine="jev") == [(0.8, False)]
    assert journal.labeled_pairs("fit", "noul_p", engine="jev") == [(0.7, False)]
    assert journal.labeled_pairs("gate", "noul_p", engine="jev") == [(0.6, False)]


async def test_record_action_without_label_keeps_old_single_key_behaviour(tmp_path, monkeypatch) -> None:
    journal = _journal(tmp_path, monkeypatch)
    journal.record_decision(**_decision_row("cid-old"))

    outcomes.record_action(call_id="cid-old", key="pick", response={"status": "ok"})

    assert journal.labeled_pairs("pick", "confidence", engine="jev") == [(0.8, True)]
    # no label -> only the single `key` is tested, fit/gate stay unlabeled
    assert journal.labeled_pairs("fit", "noul_p", engine="jev") == []
    assert journal.labeled_pairs("gate", "noul_p", engine="jev") == []
