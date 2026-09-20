"""Jev client: typed questions in, typed calibrated answers out.

Per TypeSafe's docs (docs.typesafe.ai/api, /primitives/{choice,noul,score}):
a request evaluates one `state` against a map of named `questions`; independent
questions run in the same request and cannot see each other's answers. Three
question types: `noul` (yes/no probability, no confidence), `choice` (pick one,
criteria is a map of option->description, returns probabilities+confidence),
`score` (criteria is an ORDERED LIST of level descriptions, not a map).

Calls TypeSafe's own API directly: `POST https://api.typesafe.ai/v1/systemone`,
same `state`/`questions` body shape either way, answers wrapped under `answers`.
Model is pinned to a specific version, not the `-latest` alias, since gate.py's
threshold was calibrated against one version's confidence computation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Literal

import httpx
from pydantic import BaseModel

from . import decision_log, shadow
from .ledger import EngineInfo, Usage, UsageLedger

DECISIONS_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"
RETRY_STATUS_CODES = {429, 529}
MAX_RETRIES = 3


class Noul(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: str
    criteria: dict[str, str] | None = None


class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str | None]


class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


Question = Noul | Choice | Score


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str] | None = None
    probabilities: dict[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class JevError(RuntimeError):
    pass


def _parse_answer(raw: dict) -> Answer:
    kind = raw["type"]
    if kind == "noul":
        return NoulAnswer.model_validate(raw)
    if kind == "choice":
        return ChoiceAnswer.model_validate(raw)
    if kind == "score":
        return ScoreAnswer.model_validate(raw)
    raise JevError(f"unknown answer type: {kind!r}")


def _safe_error_summary(error: Exception) -> str:
    response = getattr(error, "response", None)
    if response is None:
        return type(error).__name__
    detail = response.text.strip()[:300]
    return f"{type(error).__name__}, status={response.status_code}" + (f": {detail}" if detail else "")


class JevClient:
    """Batches independent questions over one state into a single request.
    Reuses one httpx client across calls instead of paying a fresh TLS
    handshake per question."""

    def __init__(
        self, api_key: str | None, model: str = DEFAULT_MODEL,
        *, journal: decision_log.DecisionJournal | None = None,
    ) -> None:
        self._api_key = api_key
        self._engine = "jev"  # journal rows; the LayaClient twin overrides this and reuses ask() unchanged
        self._model = model
        self._client = httpx.AsyncClient(timeout=30)
        self._journal = journal  # None -> module-level default journal (decision_log.get_journal())
        self.usage = UsageLedger()
        # Journal rows carry engine+model_revision already; the ledger keeps the
        # same identity so token snapshots from both engines distinguish themselves.
        self.usage.record_engine(EngineInfo(engine="jev", model_revision=model))
        # P5 shadow mode (shadow.py): a second engine observing this client's
        # asks. None here -- common.bootstrap() attaches via shadow.attach();
        # tests construct clients shadow-free.
        self.shadow = None
        self.shadow_tasks: set[asyncio.Task] = set()  # in-flight shadow asks; drained by aclose()

    @property
    def engine_name(self) -> str:
        """Which engine answers ask() -- budget profiles key off this, and both
        engine names can coexist in one process."""
        return self._engine

    async def aclose(self) -> None:
        await shadow.drain(self)  # keep short shadow predicts, cancel cold loads
        if self.shadow is not None:
            await self.shadow.aclose()
        await self._client.aclose()

    async def ask(
        self, state: object, questions: dict[str, Question], *,
        phase: str | None = None, goal_id: str | None = None,
        call_id: str | None = None, truncation: dict | None = None,
    ) -> dict[str, Answer]:
        """Wire contract (FROZEN): the JSON body is exactly {model, state, questions}.
        phase/goal_id/call_id/truncation are keyword-only, wire-invisible journaling
        metadata (Standing rule 4). call_id links a decision row to its downstream
        outcome row; callers that need the linkage (the gate -> approval flow) supply
        one, every other call site gets a generated one. goal_id falls back to the
        active goal_scope() so deep call sites don't thread it."""
        if not self._api_key:
            raise JevError("TYPESAFE_AI_API is not configured")
        if not questions:
            raise JevError("ask() requires at least one question")
        call_id = call_id or str(uuid.uuid4())
        scope_goal_id, scope_goal = decision_log.current_goal()
        usage_before = self.usage.snapshot()
        started = time.perf_counter()
        wire_questions = {name: q.model_dump(mode="json", exclude_none=True) for name, q in questions.items()}
        # Shadow observation (P5): the second engine sees the same state and the
        # SAME Question objects (it serializes them itself, identically), links
        # its row back via shadow_of, and its answers go nowhere.
        shadow.schedule(self, point="before_request", primary_call_id=call_id, state=state,
                        questions=questions, phase=phase, goal_id=goal_id or scope_goal_id,
                        truncation=truncation)
        body = {
            "model": self._model,
            "state": state,
            "questions": wire_questions,
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        answers: dict[str, Answer] | None = None
        error: str | None = None
        try:
            for attempt in range(MAX_RETRIES):
                try:
                    response = await self._client.post(DECISIONS_URL, headers=headers, json=body)
                    response.raise_for_status()
                    break
                except httpx.HTTPStatusError as status_error:
                    if status_error.response.status_code not in RETRY_STATUS_CODES or attempt == MAX_RETRIES - 1:
                        raise JevError(f"Jev request failed ({_safe_error_summary(status_error)})") from status_error
                    await asyncio.sleep(2**attempt)
                except httpx.HTTPError as request_error:
                    raise JevError(f"Jev request failed ({_safe_error_summary(request_error)})") from request_error
            payload = response.json()
            if usage := payload.get("usage"):
                # Only the fields the ledger tracks -- an API that adds or drops a key must
                # not turn an otherwise-successful request into a TypeError.
                self.usage.record(Usage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                ))
            if "answers" not in payload:
                raise JevError(f"Jev response has no answers: {str(payload)[:300]}")
            answers = {name: _parse_answer(raw) for name, raw in payload["answers"].items()}
        except JevError as caught:
            error = str(caught)
            raise
        except Exception as caught:  # journal every failure path, not just JevError
            error = f"{type(caught).__name__}: {caught}"
            raise
        finally:
            usage_after = self.usage.snapshot()
            self._emit_decision_row(
                call_id=call_id, phase=phase, state=state, questions=wire_questions,
                answers=answers, error=error, truncation=truncation,
                goal_id=goal_id or scope_goal_id, goal=scope_goal,
                usage_delta={
                    "input_tokens": usage_after.input_tokens - usage_before.input_tokens,
                    "output_tokens": usage_after.output_tokens - usage_before.output_tokens,
                },
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            # after-mode shadowing: only now, once the primary answer and row are
            # final, does the second engine see the call (zero contention with it).
            shadow.schedule(self, point="after_answer", primary_call_id=call_id, state=state,
                            questions=questions, phase=phase, goal_id=goal_id or scope_goal_id,
                            truncation=truncation)
        return answers

    def _emit_decision_row(
        self, *, call_id: str, phase: str | None, state: object,
        questions: dict, answers: dict[str, Answer] | None, error: str | None,
        truncation: dict | None, goal_id: str | None, goal: str | None, usage_delta: dict,
        elapsed_ms: float | None = None, shadow_of: str | None = None,
    ) -> None:
        """Journal emission, fail-open: a journal failure must never break a live
        decision (DecisionJournal.record_* swallows and prints)."""
        journal = self._journal
        if journal is None:
            journal = decision_log.get_journal()
        journal.record_decision(
            call_id=call_id, engine=self._engine, model_revision=self._model,
            phase=phase, state=state, questions=questions,
            answers={name: answer.model_dump(mode="json") for name, answer in answers.items()} if answers else None,
            error=error, truncation=truncation, goal_id=goal_id, goal=goal, usage=usage_delta,
            elapsed_ms=elapsed_ms, shadow_of=shadow_of,
        )
