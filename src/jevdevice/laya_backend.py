"""Laya backend: the ask contract answered in-process by one pinned
checkpoint (selected by JEV_ENGINE=laya) -- a plain typesymbolic `JudgeEngine`,
duck-type twin of `JevEngine`, so everything upstream of `jev.ask()` is
engine-agnostic. Differences from the Jev path, all deliberate:

- No HTTP: judge calls are local (~65 ms warm), so there is nothing to retry
  and aclose() is a no-op.
- The checkpoint is pinned to ONE HF revision (REVISION below): gate
  thresholds are calibrated against exactly that checkpoint's probability
  shape, and recalibration and any fine-tune must both anchor to a single,
  nameable artifact.
- Construction form is laya's real API: Router(preload=True, model=...) does
  not exist. Router() + .load("typed-decisions") is the verified form; NEVER
  Router(preload=["typed-decisions"]) -- __init__ calls self.preload() bare
  and loads ALL THREE checkpoints (~1.16B params), which OOMs a 4 GiB GPU
  into silent CPU fallback.

Revision pinning is done here, not by laya: laya's own snapshot_download call
takes no revision argument, so this module resolves the exact commit itself
and hands the local directory to Agent, registering it via router.attach so
predict(model="typed-decisions") routes to the pinned agent.
"""

from __future__ import annotations

import asyncio
import os
import time

from typesymbolic.judge import AskResult, JudgeError
from typesymbolic.question import Answer, Question

from .ledger import EngineInfo, UsageLedger

REPO = "convaiinnovations/laya"  # bundled repo; typed-decisions is a subfolder
MODEL = "typed-decisions"
# The snapshot commit that was on disk and answering when the engine facts were
# first verified. Calibration ties to this artifact; bumping it invalidates
# every fitted threshold until recalibration reruns.
REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
# A fine-tuned successor checkpoint (trained on the journal export, pushed and
# pinned on the hub) is selected here -- a named knob, not a code edit. Every
# decision row and ledger entry carries whichever revision actually answered,
# so A/B against the base is a config comparison, never an archaeology dig.
# Thresholds are calibrated per revision: switching this knob invalidates the
# fitted thresholds until the recalibration reruns against the new artifact.
ENV_REVISION = "JEV_LAYA_REVISION"


def active_revision() -> str:
    return os.environ.get(ENV_REVISION) or REVISION


class LayaClient:
    """A typesymbolic `JudgeEngine`: `name`/`ask_all()` only -- no journal
    emission of its own (jev.ask() does that for every engine alike). The
    checkpoint loads lazily on the first ask_all() (once per process -- a
    resident load, not a per-call cost; the snapshot itself resolves from the
    local HF cache offline) inside a worker thread so the event loop never
    blocks. Concurrent asks serialize on one lock: laya's Agent is not
    documented as thread-safe, and one forward pass is shorter than any retry
    would be.
    """

    name = "laya"

    def __init__(self, *, device: str | None = None, router: object | None = None) -> None:
        self._model = active_revision()  # journal model_revision: the pinned checkpoint
        self.usage = UsageLedger()
        self.usage.record_engine(EngineInfo(engine="laya", model_revision=self._model, model=MODEL))
        self._device = device
        self._router = router
        self._load_lock = asyncio.Lock()
        self._ask_lock = asyncio.Lock()

    async def aclose(self) -> None:
        """Nothing to release: no network client; the checkpoint stays resident
        for the process lifetime (an unload knob, if ever needed, would live here)."""

    def _build_router(self):
        """laya imports here, not at module top: the jev path must not pay the
        torch import on every startup (mcp_server imports this module via
        common.bootstrap)."""
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from laya import Router
        from laya.agent import Agent

        # Offline-first: the revision is pinned, so once the snapshot is in the
        # local HF cache every later load resolves from disk -- no network
        # round trip, no re-download check, on every process start.
        try:
            location = snapshot_download(REPO, revision=self._model, allow_patterns=[f"{MODEL}/*"], local_files_only=True)
        except LocalEntryNotFoundError:
            # First run on this machine (or the cache was pruned): fetch the
            # pinned revision once, after which loads stay offline.
            location = snapshot_download(REPO, revision=self._model, allow_patterns=[f"{MODEL}/*"])
        agent = Agent(location, subfolder=MODEL, device=self._device)
        router = Router()
        # attach, NOT Router(preload=[...]) -- that form loads all three
        # checkpoints and OOMs a 4 GiB GPU into silent CPU fallback.
        router.attach(MODEL, agent)
        return router

    async def _router_ready(self):
        if self._router is None:
            async with self._load_lock:
                if self._router is None:  # double-checked: exactly one build
                    self._router = await asyncio.to_thread(self._build_router)
        return self._router

    async def ask_all(self, state: dict, questions: dict[str, Question]) -> AskResult:
        """Budget overflow -- laya raises ValueError when cfg max_len cuts below
        the option block -- surfaces as JudgeError like every other engine
        failure; the message may blame head_max_len even when max_len was the
        trigger, so catch the type, never parse the text."""
        if not questions:
            raise JudgeError("ask_all() requires at least one question")
        wire_questions = {name: q.model_dump(mode="json", exclude_none=True) for name, q in questions.items()}
        router = await self._router_ready()
        started = time.perf_counter()
        try:
            # One forward pass at a time: the worker-thread predict must not
            # overlap itself on the shared Agent.
            async with self._ask_lock:
                payload = await asyncio.to_thread(router.predict, state, wire_questions, MODEL)
        except ValueError as budget_error:
            raise JudgeError(f"Laya budget overflow: {budget_error}") from budget_error
        elapsed_ms = (time.perf_counter() - started) * 1000
        usage = payload.get("usage")
        # Per-ask routing evidence (Router.predict embeds its RouteDecision);
        # explicit-model routing is constant, but the ledger keeps the proof.
        self.usage.record_engine(EngineInfo(
            engine="laya", model_revision=self._model, model=MODEL, routing=payload.get("routing"),
        ))
        if "answers" not in payload:
            raise JudgeError(f"Laya response has no answers: {str(payload)[:300]}")
        answers = {name: Answer.model_validate({**raw, "qid": name})
                  for name, raw in payload["answers"].items()}
        return AskResult(answers=answers, model_revision=self._model, usage=usage, elapsed_ms=elapsed_ms)
