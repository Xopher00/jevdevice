"""P7.5 T4: per-tier telemetry from the journal -- journal-only, zero model
calls, zero device. Every planner row carries additive `tier`/`recipe_id`
fields, so the decomposition profile is a journal query.

Output: eval/phase75_recipes/tier_report.json + stdout JSON:
  per tier {0..4}: resolutions, fall-throughs, plus the step outcomes
  (verified/failed/escalated/none/skipped) attributed to that tier's asks and
  executions.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from jevdevice.decision_log import DecisionJournal

OUT_DIR = REPO / "eval" / "phase75_recipes"
PLANNER_STATUSES = {"planner_resolved", "planner_fallthrough", "cold_goal", "human_escalation"}


def report(journal) -> dict:
    tier_events: dict[int, Counter] = {tier: Counter() for tier in range(5)}
    step_outcomes: dict[int, Counter] = {tier: Counter() for tier in range(5)}
    recipe_ids: set[str] = set()
    goals_seen: set[tuple] = set()
    for row in journal.replay():
        if row.get("type") != "outcome":
            continue
        tier = row.get("tier")
        status = row.get("status")
        if tier is None and status not in PLANNER_STATUSES:
            continue  # an ordinary action row (tier absent = not planner-driven)
        if tier is None:
            tier = 0
        recipe_ids.add(row.get("recipe_id")) if row.get("recipe_id") else None
        if status in PLANNER_STATUSES:
            tier_events[tier][status] += 1
            if row.get("goal"):
                goals_seen.add((row["goal"], status))
        elif row.get("kind"):
            step_outcomes[tier][row.get("verification") or "none"] += 1
    return {
        "tiers": {
            str(tier): {
                "planner_resolved": tier_events[tier]["planner_resolved"],
                "planner_fallthrough": tier_events[tier]["planner_fallthrough"],
                "cold_goal": tier_events[tier]["cold_goal"],
                "human_escalation": tier_events[tier]["human_escalation"],
                "step_outcomes": dict(step_outcomes[tier]),
            }
            for tier in range(5)
        },
        "recipes_replayed": sorted(recipe_ids),
        "goals_seen": len(goals_seen),
    }


def main() -> int:
    out = report(DecisionJournal())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tier_report.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
