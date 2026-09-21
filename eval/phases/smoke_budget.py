"""Live smoke: the in-process engine's budget-profile path on a real device.

launch_app_for_goal directly on dev goals (pick_kind's own gate escalates
separately) -- shows the retrieval shortlist, the chunked-sweep fallback,
and the journal rows with option counts.

Run: ANDROID_SERIAL=<serial> JEV_ENGINE=laya uv run python eval/phases/smoke_budget.py
"""

from __future__ import annotations

import asyncio
import json

from jevdevice.actions.app_launch import launch_app_for_goal
from jevdevice.budget import LAYA_PROFILE, NONE_OF_THESE
from jevdevice.common import bootstrap
from jevdevice.journal.decision_log import DecisionJournal, goal_id_for, goal_scope

# Dev-split goals only (eval/goals.yaml); the held-out half is untouched.
GOALS = [
    "Open the Calculator app.",   # pa01 -- should resolve via one choice call
    "Open the Settings app.",     # also a real app; second data point
]


def _criteria_len(row: dict) -> int:
    questions = row.get("questions") or {}
    for q in questions.values():
        criteria = (q.get("criteria") if isinstance(q, dict) else None) or {}
        if NONE_OF_THESE in criteria:
            return len(criteria)
    return 0


async def main() -> None:
    jev, transport = bootstrap()
    assert LAYA_PROFILE.engine == "laya"
    for goal in GOALS:
        print(f"=== goal: {goal!r} ===", flush=True)
        with goal_scope(goal):
            outcome = await launch_app_for_goal(jev, transport, goal)
        print(json.dumps({
            "goal": goal, "package": outcome.package, "launched": outcome.launched,
            "confidence": round(outcome.confidence, 2), "satisfied": round(outcome.satisfied, 2),
            "attempts": outcome.attempts, "observed": outcome.observed,
            "reasons": list(outcome.reasons),
        }), flush=True)
    print("=== JOURNAL ROWS (this session's goals) ===")
    ids = {goal_id_for(goal) for goal in GOALS}
    for row in DecisionJournal().replay():
        if row.get("goal_id") in ids and row.get("type") == "decision":
            print(json.dumps({
                "engine": row.get("engine"), "phase": row.get("phase"),
                "options_in_choice": _criteria_len(row),
                "error": row.get("error"),
                "usage": row.get("usage"),
            }))


if __name__ == "__main__":
    asyncio.run(main())
