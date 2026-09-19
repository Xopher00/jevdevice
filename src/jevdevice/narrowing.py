"""Two-round narrowing over a real, runtime-discovered candidate set, replacing
a lexical pre-filter (matching.fuzzy_narrow) with Jev's own semantic judgment,
so a goal with no string overlap with the right answer can still reach it.

Round 1 chunks the real candidates (never drops one — matching.chunk_candidates
only orders them) and asks, per chunk, "is anything here relevant" plus a
Choice over that chunk's real members, run concurrently since each chunk needs
its own request (a Choice's criteria can't span chunks without recreating the
255-option problem chunking exists to avoid). Round 2 re-ranks the merged
shortlist with one Choice and one "does this really fit" Noul per entry, and
matching.decide() combines everything.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from .jev import Choice, JevClient, Noul
from .matching import NarrowVerdict, chunk_candidates, decide, ground_candidates

CHUNK_SIZE = 200  # headroom under Jev's 255-option Choice cap
BEAM_K = 3  # hierarchical_classification.md's measured beam width (greedy K=1 matched 2/4, K=3 matched 4/4)
MAX_CONCURRENT = 3  # bound the fan-out; jev.py already backs off on 429/529


def _as_criteria(items: list[str]) -> dict[str, None]:
    return {item: None for item in items}


async def _score_chunk(
    jev: JevClient, query: str, chunk: list[str], instructions: str, state_extra: dict | None,
) -> tuple[float, dict[str, float]]:
    """One ask per chunk: is anything here relevant, plus a Choice over this
    chunk's real members. Returns (chunk_relevance, {candidate: probability})."""
    if not chunk:
        return 0.0, {}
    state = {"goal": query, "candidates": chunk, **(state_extra or {})}
    answers = await jev.ask(state, {
        "any": Noul(instructions="Could any of these candidates satisfy the goal?"),
        "pick": Choice(instructions=instructions, criteria=_as_criteria(chunk)),
    })
    return answers["any"].noul, answers["pick"].probabilities


async def semantic_shortlist(
    jev: JevClient, query: str, candidates: list[str], *, instructions: str,
    chunk_size: int = CHUNK_SIZE, k: int = BEAM_K, state_extra: dict | None = None,
) -> list[str]:
    """Round 1: chunk relevance * within-chunk probability, keep the top k
    per chunk (beam, not greedy top-1), merged into one shortlist."""
    chunks = chunk_candidates(candidates, query, chunk_size)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def bounded(chunk: list[str]) -> tuple[float, dict[str, float]]:
        async with semaphore:
            return await _score_chunk(jev, query, chunk, instructions, state_extra)

    results = await asyncio.gather(*(bounded(chunk) for chunk in chunks))

    scored: list[tuple[float, str]] = []
    for relevance, probabilities in results:
        top_k = sorted(probabilities.items(), key=lambda kv: -kv[1])[:k]
        scored.extend((relevance * probability, candidate) for candidate, probability in top_k)
    scored.sort(key=lambda sp: -sp[0])
    return [candidate for _, candidate in scored[:k * len(chunks)]]


async def narrow_and_pick(
    jev: JevClient, query: str, candidates: list[str], *, instructions: str, fit_instructions: str,
    evidence_for: Callable[[list[str]], Awaitable[dict[str, str]]] | None = None,
    chunk_size: int = CHUNK_SIZE, k: int = BEAM_K, min_fit: float = 0.5, min_confidence: float = 0.6,
    min_margin: float = 0.15, state_extra: dict | None = None, accept_any_fitting: bool = False,
    describe: Callable[[str], str] = lambda c: c,
) -> NarrowVerdict:
    """Rounds 1+2+decide: the one function real call sites use.

    Round 1 exists to reduce a real enumeration too large for one Choice
    (packages, services) down to a shortlist. When candidates already fit in
    one chunk (on-screen elements, always small), round 1 buys nothing but a
    round trip and a beam_k=3 cutoff that can lose the right answer among
    otherwise-few real options -- skip straight to round 2 with everything."""
    if len(candidates) <= chunk_size:
        shortlist = candidates
    else:
        shortlist, _ = ground_candidates(
            await semantic_shortlist(jev, query, candidates, instructions=instructions,
                                      chunk_size=chunk_size, k=k, state_extra=state_extra),
            candidates,
        )
    if not shortlist:
        return decide(None, {}, 0.0, {}, candidates, min_fit=min_fit, min_confidence=min_confidence, min_margin=min_margin, accept_any_fitting=accept_any_fitting)

    evidence = await evidence_for(shortlist) if evidence_for else {}
    criteria = {c: evidence.get(c) for c in shortlist}
    fit_keys = {f"fit_{i}": c for i, c in enumerate(shortlist)}
    state = {"goal": query, "candidates": criteria, **(state_extra or {})}
    answers = await jev.ask(state, {
        "pick": Choice(instructions=instructions, criteria=criteria),
        **{key: Noul(instructions=fit_instructions.format(candidate=describe(c))) for key, c in fit_keys.items()},
    })
    pick = answers["pick"]
    fits = {c: answers[key].noul for key, c in fit_keys.items()}
    return decide(pick.choice, pick.probabilities, pick.confidence, fits, candidates, min_fit=min_fit, min_confidence=min_confidence, min_margin=min_margin, accept_any_fitting=accept_any_fitting)
