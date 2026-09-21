"""METR Task Standard-shaped adapter for the phone task family.

The engine never imports this: it maps the runtime's existing surfaces onto the
METR task-family vocabulary so a second family (P10) swaps the adapter, not the
engine.

- `get_tasks()` -> the DEV-half goal ids of eval/goals.yaml (the held-out half
  is sacred and never exposed; splitguard enforces it).
- `add_instructions(task)` -> the goal text (what device_do already consumes).
- `verify(task)` -> the runtime's own verification, replayed from the journal:
  device-VERIFIED outcome rows joined by goal_id (zero model calls -- the same
  join the recovery miner and recipe builder use). No separate verifier is
  invented; the runtime already verifies against device truth.

Task repos stay private: goals.yaml and the journal live in this repo and are
never published (contamination guardrail).
"""

from __future__ import annotations

import sys
from pathlib import Path

PHASES = Path(__file__).resolve().parent
if str(PHASES) not in sys.path:
    sys.path.insert(0, str(PHASES))

from splitguard import dev_goals

from jevdevice.journal.decision_log import VERIFIED, goal_id_for

FAMILY_NAME = "jevdevice-phone"
# METR task families expose one family per deployment surface; the phone goals
# carry both question-style and action-style tasks in goals.yaml.
GOAL_FAMILIES = ("phone_questions", "phone_actions")


def dev_phone_entries() -> list[dict]:
    """Dev-split entries of both phone families (questions + actions)."""
    return [entry for family in GOAL_FAMILIES for entry in dev_goals(family)]


class PhoneTaskFamily:
    """The Android phone goals in the METR task-family shape. Ours are goals
    against a live device, so there is no per-task environment to boot (the
    device state IS the task state) and scoring is the journal's verified
    outcome -- the same verification run_kind already enforces."""

    def __init__(self) -> None:
        self._goals = dev_phone_entries()
        self._tasks = {goal_id_for(e["goal"]): e["goal"] for e in self._goals}

    def get_tasks(self) -> dict[str, str]:
        """METR shape: task name -> task. Ours: goal_id -> goal text (dev half only)."""
        return dict(self._tasks)

    def add_instructions(self, task_name: str) -> str:
        """The agent-facing instructions: the frozen goal text, verbatim."""
        if task_name not in self._tasks:
            raise KeyError(f"unknown task {task_name!r} (dev-half tasks only)")
        return self._tasks[task_name]

    async def run_task(self, task_name: str, jev, device, *, executor=None, store=None):
        """One task through the real runtime: the tiered planner resolves it
        (recipe hit -> adapt -> stepwise -> cold goal), never auto-approving.
        Returns the PlannerResult; scoring is `verify`."""
        from jevdevice.execution.planner import resolve

        return await resolve(jev, device, self.add_instructions(task_name),
                             executor=executor, store=store)

    def verify(self, task_name: str, journal_rows=None) -> float:
        """Device-truth score in [0, 1] from the journal: 1.0 iff the task's
        goal_id has at least one device-VERIFIED outcome row (the same
        verification outcomes.emit_outcome records). Journal-only: zero model
        calls, no device round trip."""
        rows = [r for r in (journal_rows if journal_rows is not None else _journal_rows())
                if r.get("goal_id") == task_name]
        return 1.0 if any(r.get("verification") == VERIFIED for r in rows) else 0.0


def aggregate_scores(task_names: list[str], journal_rows=None) -> dict[str, float]:
    """METR's aggregate hook over a run's tasks: per-task verify scores."""
    family = PhoneTaskFamily()
    return {name: family.verify(name, journal_rows) for name in task_names}


def _journal_rows() -> list[dict]:
    from jevdevice.journal.decision_log import get_journal

    return list(get_journal().replay())