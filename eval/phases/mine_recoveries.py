"""Failure mining: an escalation is an off-trajectory state, and a recovery
that completes the goal afterwards is a (broken state → recovery action)
pair. Training loses on the recovery actions only; the broken prefix is
context, not supervision.

Offline and journal-only (zero model calls):

  - trajectories = primary journal rows grouped by goal_id, ts-ordered
  - failure = an outcome row with verification escalated/failed/none
  - recovery = the NEXT outcome row for the SAME goal (same session window)
    whose verification is verified — provenance "human_corrected" when a
    device_approve recovery_command ran, "retry" when a fresh attempt succeeded
  - every pair references the failed trajectory by call_id chain

Dev-half only: goal texts are asserted absent from the held-out half (the
harness only ever runs dev goals; this is a second guard, not a license).

  uv run python eval/phases/mine_recoveries.py [journal_dir]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

from jevdevice.journal.decision_log import (
    ESCALATED,
    FAILED,
    NONE,
    VERIFIED,
    DecisionJournal,
)

OUT_FILE = REPO / "eval" / "phases" / "flywheel" / "recovery_pairs.jsonl"
MAX_CHAIN_CALL_IDS = 20  # per side of the pair; bounded, named knob

FAILURE_VERIFICATIONS = frozenset({ESCALATED, FAILED, NONE})


def heldout_goals() -> set[str]:
    """Held-out goal texts (casefolded for matching), from the frozen split."""
    import splitguard

    return {e["goal"].casefold() for e in splitguard.heldout_goals()}


def _ts(row: dict) -> str:
    return row.get("ts") or ""


def trajectories_by_goal(journal: DecisionJournal) -> dict[str, list[dict]]:
    """Primary (shadow-free) decision+outcome rows grouped by goal_id, ordered.
    Ordered by (ts, write order): same-millisecond rows keep their on-disk order,
    so a failure and its recovery never compare equal."""
    grouped: dict[str, list[tuple[tuple[str, int], dict]]] = {}
    for path in sorted(journal.directory.glob("journal-*.jsonl")):
        for seq, row in enumerate(journal.replay(day=path.stem.removeprefix("journal-"))):
            if row.get("type") not in ("decision", "outcome") or row.get("shadow_of"):
                continue
            goal_id = row.get("goal_id")
            if goal_id:
                grouped.setdefault(goal_id, []).append(((_ts(row), seq), row))
    return {goal_id: [row for _, row in sorted(entries, key=lambda item: item[0])]
            for goal_id, entries in grouped.items()}


def mine_pair(rows: list[dict]) -> dict | None:
    """The first (failure → verified recovery) chain in one goal's trajectory,
    referencing the failed side by call_id chain. None when nothing failed.
    Rows arrive ts-ordered (ties keep write order), so position is the clock."""
    outcomes = [r for r in rows if r.get("type") == "outcome"]
    failures = [r for r in outcomes if r.get("verification") in FAILURE_VERIFICATIONS]
    if not failures:
        return None
    failure = failures[0]
    failure_index = rows.index(failure)
    recovery = next(
        (r for r in rows[failure_index + 1:]
         if r.get("type") == "outcome" and r.get("verification") == VERIFIED),
        None,
    )
    if recovery is None:
        return None
    recovery_index = rows.index(recovery)
    failed_side_ids = [r["call_id"] for r in rows[:failure_index + 1]
                       if r.get("type") == "decision" and r.get("call_id")]
    recovery_side_ids = [r["call_id"] for r in rows[failure_index + 1:recovery_index + 1]
                         if r.get("type") == "decision" and r.get("call_id")]
    provenance = "human_corrected" if recovery.get("recovery_command") else "retry"
    return {
        "goal_id": failure.get("goal_id"),
        "goal": failure.get("goal"),
        "broken": {
            "ts": failure.get("ts"),
            "call_id": failure.get("call_id"),
            "status": failure.get("status"),
            "reasons": failure.get("reasons"),
            "attempted_command": failure.get("executed_command"),
            "failed_trajectory_call_ids": failed_side_ids[-MAX_CHAIN_CALL_IDS:],
        },
        "recovery": {
            "ts": recovery.get("ts"),
            "call_id": recovery.get("call_id"),
            "executed_command": recovery.get("executed_command"),
            "recovery_command": recovery.get("recovery_command"),
            "recovery_decision_call_ids": recovery_side_ids[-MAX_CHAIN_CALL_IDS:],
            "device": recovery.get("device"),
        },
        "provenance": provenance,
    }


def collect(journal: DecisionJournal) -> tuple[list[dict], Counter]:
    heldout = heldout_goals()
    dropped: Counter = Counter()
    pairs: list[dict] = []
    for goal_id, rows in trajectories_by_goal(journal).items():
        goal_text = next((r.get("goal") for r in rows if r.get("goal")), "")
        if goal_text.casefold() in heldout:
            dropped["heldout_goal"] += 1
            continue
        pair = mine_pair(rows)
        if pair is None:
            dropped["no_failure_or_no_recovery"] += 1
            continue
        pair["goal_id"] = goal_id
        pair["goal"] = goal_text
        pairs.append(pair)
    return pairs, dropped


def main() -> int:
    journal = DecisionJournal(sys.argv[1] if len(sys.argv) > 1 else None)
    pairs, dropped = collect(journal)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(pair, separators=(",", ":"), default=str) + "\n" for pair in pairs)
    summary = {
        "recovery_pairs": len(pairs),
        "provenance": dict(Counter(pair["provenance"] for pair in pairs)),
        "goals_with_failure_unrecovered": dropped.get("no_failure_or_no_recovery", 0),
        "heldout_goals_skipped": dropped.get("heldout_goal", 0),
        "out_file": str(OUT_FILE),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
