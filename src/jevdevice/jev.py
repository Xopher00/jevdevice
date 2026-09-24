"""jevdevice's one binding onto core's judge/ask boundary: a single
`ask()` that journals goal scope + truncation/escape-hatch provenance and
feeds the calling engine's UsageLedger. Transport, retry, and answer parsing
all live in typesymbolic now (`JevEngine`); nothing here talks HTTP.
"""

from __future__ import annotations

from typesymbolic.domain import Facts
from typesymbolic.engine import ask_batch
from typesymbolic.judge import AskResult, JudgeEngine, JudgeError
from typesymbolic.question import Answer, Choice, Noul, Question, Score

from jevdevice.journal import decision_log

from .judge import shadow
from .ledger import Usage

__all__ = ["Answer", "AskResult", "Choice", "JudgeEngine", "JudgeError", "Noul", "Question", "Score", "ask"]


async def ask(
    judge: JudgeEngine, state: object, questions: dict[str, Question], *,
    phase: str | None = None, truncation: dict | None = None, shadow_of: str | None = None,
) -> tuple[str, dict[str, Answer]]:
    """One batched ask, journaled under a fresh call_id. goal/goal_id/shadow_of
    ride `scope` (wire-invisible); truncation and escape-hatch generated-source
    ride `extra`. `shadow_of` is set only by judge/shadow.py's own re-ask."""
    from .question_sets import generated_source

    goal_id, goal = decision_log.current_goal()
    scope = {k: v for k, v in (("goal_id", goal_id), ("goal", goal), ("shadow_of", shadow_of)) if v is not None}
    extra = {k: v for k, v in (("truncation", truncation), ("generated", generated_source(questions))) if v is not None}
    call_id, result = await ask_batch(
        judge=judge, questions=questions, facts=Facts(state=state),
        journal=decision_log.get_journal(), phase=phase, scope=scope or None,
        capture=True, extra=extra or None,
    )
    if shadow_of is None:  # the shadow's own re-ask never spawns a shadow of itself
        shadow.schedule(judge, primary_call_id=call_id, state=state, questions=questions,
                        phase=phase, truncation=truncation)
    ledger = getattr(judge, "usage", None)
    if ledger is not None and result.usage:
        ledger.record(Usage(**result.usage))
    return call_id, result.answers
