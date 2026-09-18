"""Jev client: typed questions in, typed calibrated answers out.

Per TypeSafe's docs (docs.typesafe.ai/api, /primitives/{choice,noul,score}):
a request evaluates one `state` against a map of named `questions`; independent
questions run in the same request and cannot see each other's answers. Three
question types: `noul` (yes/no probability, no confidence), `choice` (pick one,
criteria is a map of option->description, returns probabilities+confidence),
`score` (criteria is an ORDERED LIST of level descriptions, not a map).

The native API is `POST https://api.typesafe.ai/v1/systemone` with a native
TypeSafe key. We don't have one; OpenRouter exposes Jev at a different,
dedicated endpoint instead (confirmed live 2026-09-18): Jev models 400 on
`/chat/completions` ("is a decisions model"), and `/v1/systemone` doesn't exist
on openrouter.ai. The working route is `POST /api/alpha/decisions` with an
OpenRouter key, same `state`/`questions` body shape, response wraps answers
under `answers` exactly as documented.
"""

from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"


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
    """Batches independent questions over one state into a single request."""

    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL) -> None:
        self._api_key = api_key
        self._model = model

    def readiness_error(self) -> str | None:
        return None if self._api_key else "OPENROUTER_API_KEY is not configured"

    async def ask(self, state: object, questions: dict[str, Question]) -> dict[str, Answer]:
        if not self._api_key:
            raise JevError("OPENROUTER_API_KEY is not configured")
        if not questions:
            raise JevError("ask() requires at least one question")
        body = {
            "model": self._model,
            "state": state,
            "questions": {name: q.model_dump(mode="json", exclude_none=True) for name, q in questions.items()},
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    DECISIONS_URL,
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    json=body,
                )
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise JevError(f"Jev request failed ({_safe_error_summary(error)})") from error
        payload = response.json()
        return {name: _parse_answer(raw) for name, raw in payload["answers"].items()}
