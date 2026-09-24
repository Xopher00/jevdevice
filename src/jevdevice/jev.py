"""jevdevice's one binding onto core's judge/ask boundary: a single
`ask()` that journals goal scope + truncation/escape-hatch provenance and
feeds the calling engine's UsageLedger. Transport, retry, and answer parsing
all live in typesymbolic now (`JevEngine`); nothing here talks HTTP.
"""

from __future__ import annotations

import uuid

from typesymbolic.domain import Facts
from typesymbolic.engine import ask_batch
from typesymbolic.judge import AskResult, JudgeEngine, JudgeError
from typesymbolic.question import Answer, Choice, Noul, Question, QuestionRef, Score
from typesymbolic.vocab import calibration_unit

from jevdevice.journal import decision_log

from .judge import shadow
from .ledger import Usage

__all__ = ["Answer", "AskResult", "Choice", "JudgeEngine", "JudgeError", "Noul", "Question", "Score", "ask"]


def _refs_for(questions: dict[str, Question]) -> dict[str, QuestionRef] | None:
    """QuestionRefs for question_sets-tagged keys; untagged questions skip."""
    from .question_sets import load as load_question_set

    refs: dict[str, QuestionRef] = {}
    for key, q in questions.items():
        qid, version = getattr(q, "qid", None), getattr(q, "vocab_version", None) or None
        if qid:
            group, scale = calibration_unit(load_question_set(version), qid, q)
            refs[key] = QuestionRef(qid=qid, vocab_version=version, group=group, scale=scale)
    return refs or None


async def ask(
    judge: JudgeEngine, state: object, questions: dict[str, Question], *,
    phase: str | None = None, truncation: dict | None = None,
    call_id: str | None = None, shadow_of: str | None = None,
) -> tuple[str, dict[str, Answer]]:
    """One batched ask, journaled under `call_id` (generated up front when not
    supplied, so a shadow re-ask can carry shadow_of=<this id> before the
    primary request even lands). goal/goal_id/shadow_of ride `scope`
    (wire-invisible); truncation and escape-hatch generated-source ride
    `extra`. `shadow_of` is set only by judge/shadow.py's own re-ask; only a
    primary ask (`shadow_of is None`) carries refs, so only it can label."""
    from .question_sets import generated_source

    call_id = call_id or str(uuid.uuid4())
    refs = _refs_for(questions) if shadow_of is None else None
    goal_id, goal = decision_log.current_goal()
    scope = {k: v for k, v in (("goal_id", goal_id), ("goal", goal), ("shadow_of", shadow_of)) if v is not None}
    extra = {k: v for k, v in (("truncation", truncation), ("generated", generated_source(questions))) if v is not None}
    if shadow_of is None:  # the shadow's own re-ask never spawns a shadow of itself
        shadow.schedule(judge, point="before_request", primary_call_id=call_id, state=state,
                        questions=questions, phase=phase, truncation=truncation)
    call_id, result = await ask_batch(
        judge=judge, questions=questions, facts=Facts(state=state),
        journal=decision_log.get_journal(), phase=phase, refs=refs, scope=scope or None,
        capture=True, extra=extra or None, call_id=call_id,
    )
    if shadow_of is None:
        shadow.schedule(judge, point="after_answer", primary_call_id=call_id, state=state,
                        questions=questions, phase=phase, truncation=truncation)
    ledger = getattr(judge, "usage", None)
    if ledger is not None and result.usage:
        ledger.record(Usage(**result.usage))
    return call_id, result.answers
