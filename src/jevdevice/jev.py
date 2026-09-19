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
from typing import Literal

import httpx
from pydantic import BaseModel

from .ledger import Usage, UsageLedger

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

    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL) -> None:
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(timeout=30)
        self.usage = UsageLedger()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ask(self, state: object, questions: dict[str, Question]) -> dict[str, Answer]:
        if not self._api_key:
            raise JevError("TYPESAFE_AI_API is not configured")
        if not questions:
            raise JevError("ask() requires at least one question")
        body = {
            "model": self._model,
            "state": state,
            "questions": {name: q.model_dump(mode="json", exclude_none=True) for name, q in questions.items()},
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.post(DECISIONS_URL, headers=headers, json=body)
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in RETRY_STATUS_CODES or attempt == MAX_RETRIES - 1:
                    raise JevError(f"Jev request failed ({_safe_error_summary(error)})") from error
                await asyncio.sleep(2**attempt)
            except httpx.HTTPError as error:
                raise JevError(f"Jev request failed ({_safe_error_summary(error)})") from error
        payload = response.json()
        if "usage" in payload:
            self.usage.record(Usage(**payload["usage"]))
        return {name: _parse_answer(raw) for name, raw in payload["answers"].items()}
