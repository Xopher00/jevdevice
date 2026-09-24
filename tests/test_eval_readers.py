"""Primary+shadow decision rows and a verified outcome, through a real
typesymbolic Journal in tmp: shadow_agreement pairs off `scope.shadow_of`,
mine_recoveries reads the verdict row correctly."""

from __future__ import annotations

import sys
from pathlib import Path

PHASES = Path(__file__).resolve().parent.parent / "eval" / "phases"
if str(PHASES) not in sys.path:
    sys.path.insert(0, str(PHASES))

import mine_recoveries
import shadow_agreement
from typesymbolic.domain import ActOutcome, Verdict
from typesymbolic.journal import Journal
from typesymbolic.question import Answer


def test_shadow_pairing_and_verdict_join(tmp_path: Path) -> None:
    journal = Journal(root=tmp_path, rotation="none", background_writes=False)
    pick = Answer(qid="pick", type="choice", choice="a", probabilities={"a": 0.9}, confidence=0.9)
    journal.record_decision(call_id="p1", engine="jev", phase="ground",
                            scope={"goal": "open app", "goal_id": "g1"}, answers={"pick": pick})
    journal.record_decision(call_id="s1", engine="laya", phase="ground",
                            scope={"goal": "open app", "goal_id": "g1", "shadow_of": "p1"}, answers={"pick": pick})
    journal.record_outcome(call_id="p1", gate=None, outcome=ActOutcome(succeeded=True, key="pick"),
                           extra={"goal_id": "g1", "goal": "open app"})
    journal.record_verdict(call_id="p1", verdict=Verdict(status="verified"))
    rows = list(journal.replay())
    report = shadow_agreement.build_report(rows)
    assert report["primaries_shadowed"] == 1
    assert report["shadow_rows"] == 1
    pairs, _dropped = mine_recoveries.collect(journal)
    assert pairs == []  # no failure recorded, so nothing is mined
    outcomes = [r for r in rows if r.get("type") == "outcome"]
    verdicts = {r["call_id"]: r["status"] for r in rows if r.get("type") == "verdict"}
    assert verdicts[outcomes[0]["call_id"]] == "verified"
