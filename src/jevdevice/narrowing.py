"""Two-round narrowing over a real, runtime-discovered candidate set, replacing
a lexical pre-filter (matching.fuzzy_narrow) with Jev's own semantic judgment,
so a goal with no string overlap with the right answer can still reach it.

Round 1 chunks the real candidates (never drops one — matching.chunk_candidates
only orders them) and asks, per chunk, "is anything here relevant" plus a
Choice over that chunk's real members, run concurrently since each chunk needs
its own request (a Choice's criteria can't span chunks without recreating the
option-cap problem chunking exists to avoid). Round 2 re-ranks the merged
shortlist with one Choice and one "does this really fit" Noul per entry, and
matching.decide() combines everything.

Behavior keyed off the answering engine's budget profile (budget.py -- the
hosted engine keeps today's behavior exactly):
- chunk_size comes from the profile (in-process 20, hosted 200), not a fixed
  constant -- the in-process head fits ~20 SHORT options, not 200.
- High-cardinality decisions go shortlist-first: a fuzzy-scored retrieval
  shortlist of `profile.shortlist_size` gets ONE choice call; only if the
  judge abstains or the pick fails its gates does the lossless chunked sweep
  run as fallback. The lexical probe can't see a zero-overlap match ("check my
  email" -> gm) -- the abstain + fallback is what keeps that case reachable,
  which is why shortlist_first stays off in the hosted profile (its 2-round
  sweep is calibrated).
- Every choice also offers "none_of_these" (abstain_option profiles): the
  judge can abstain, and an abstain escalates -- never guesses.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from .budget import choice_criteria, current_profile, is_abstain
from .jev import Answer, Choice, JevClient, Noul
from .matching import (
    NarrowVerdict,
    chunk_candidates,
    decide,
    fuzzy_narrow,
    ground_candidates,
)

BEAM_K = 3  # hierarchical_classification.md's measured beam width (greedy K=1 matched 2/4, K=3 matched 4/4)
MAX_CONCURRENT = 3  # bound the fan-out; jev.py already backs off on 429/529


def _as_criteria(items: list[str]) -> dict[str, None]:
    return {item: None for item in items}


def fit_questions(shortlist: Sequence[str], fit_instructions: str, describe: Callable[[str], str] = lambda c: c) -> dict[str, Noul]:
    """One "does this candidate really fit" Noul per shortlist entry, keyed fit_0..n in
    shortlist order -- shared by narrow_and_pick and ui.py's fused pick+gate asks."""
    return {f"fit_{i}": Noul(instructions=fit_instructions.format(candidate=describe(c))) for i, c in enumerate(shortlist)}


def extract_fits(answers: dict[str, Answer], shortlist: Sequence[str]) -> dict[str, float]:
    """Candidate -> its fit Noul; keys align with fit_questions."""
    return {c: answers[f"fit_{i}"].noul for i, c in enumerate(shortlist)}


def _abstain_verdict(shortlist: Sequence[str], fits: dict[str, float], confidence: float) -> NarrowVerdict:
    """The judge picked none_of_these: escalate, never guess (fail-closed).
    Its fit Nouls are kept as evidence in the verdict."""
    return NarrowVerdict(
        None, confidence, 0.0, max(fits.values(), default=0.0), list(shortlist),
        ["judge abstained: picked none_of_these, so nothing on the shortlist fits"],
    )


async def _score_chunk(
    jev: JevClient, query: str, chunk: list[str], instructions: str, state_extra: dict | None,
) -> tuple[float, dict[str, float]]:
    """One ask per chunk: is anything here relevant, plus a Choice over this
    chunk's real members. Returns (chunk_relevance, {candidate: probability}).
    An abstain pick contributes nothing to the shortlist -- none_of_these is an
    answer about the chunk, not a candidate in it."""
    if not chunk:
        return 0.0, {}
    profile = current_profile(jev.engine_name)
    state = {"goal": query, "candidates": chunk, **(state_extra or {})}
    answers = await jev.ask(state, {
        "any": Noul(instructions="Could any of these candidates satisfy the goal?"),
        "pick": Choice(instructions=instructions, criteria=choice_criteria(_as_criteria(chunk), profile)),
    }, phase="recall")
    probabilities = {
        candidate: probability for candidate, probability in answers["pick"].probabilities.items()
        if not is_abstain(candidate)
    }
    return answers["any"].noul, probabilities


async def semantic_shortlist(
    jev: JevClient, query: str, candidates: list[str], *, instructions: str,
    chunk_size: int | None = None, k: int | None = None, state_extra: dict | None = None,
) -> list[str]:
    """Round 1: chunk relevance * within-chunk probability, keep the top k
    per chunk (beam, not greedy top-1), merged into one shortlist -- capped at
    the profile's shortlist_size, because the merged result feeds ONE round-2
    Choice whose option list must fit the head budget (more candidates than
    the head carries would be silently dropped by the engine, not error).
    chunk_size/k default to the answering engine's profile knobs."""
    profile = current_profile(jev.engine_name)
    chunk_size = profile.chunk_size if chunk_size is None else chunk_size
    k = BEAM_K if k is None else k
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
    cap = min(k * len(chunks), profile.shortlist_size)
    return [candidate for _, candidate in scored[:cap]]


async def _pick_and_decide(
    jev: JevClient, query: str, shortlist: Sequence[str], *, instructions: str, fit_instructions: str,
    evidence_for: Callable[[list[str]], Awaitable[dict[str, str]]] | None,
    state_extra: dict | None, describe: Callable[[str], str],
    enumerated: Sequence[str], min_fit: float, min_confidence: float, min_margin: float,
    accept_any_fitting: bool,
) -> NarrowVerdict:
    """Round 2 shared by every path (direct, retrieval shortlist, chunked sweep):
    one Choice over the shortlist plus a fit Noul per entry, then decide()."""
    profile = current_profile(jev.engine_name)
    if not shortlist:
        return decide(None, {}, 0.0, {}, enumerated, min_fit=min_fit, min_confidence=min_confidence, min_margin=min_margin, accept_any_fitting=accept_any_fitting)
    evidence = await evidence_for(list(shortlist)) if evidence_for else {}
    # profiles with descriptions_in_state put each option's rich description in
    # the criteria VALUE -- it rides in the state body, while the option list in
    # the head stays short. The hosted engine keeps bare options.
    if profile.descriptions_in_state:
        criteria = {c: evidence.get(c) or describe(c) for c in shortlist}
    else:
        criteria = {c: evidence.get(c) for c in shortlist}
    state = {"goal": query, "candidates": criteria, **(state_extra or {})}
    answers = await jev.ask(state, {
        "pick": Choice(instructions=instructions, criteria=choice_criteria(criteria, profile)),
        **fit_questions(shortlist, fit_instructions, describe),
    }, phase="ground")
    pick = answers["pick"]
    fits = extract_fits(answers, shortlist)
    if is_abstain(pick.choice):
        return _abstain_verdict(shortlist, fits, pick.confidence)
    return decide(pick.choice, pick.probabilities, pick.confidence, fits, enumerated, min_fit=min_fit, min_confidence=min_confidence, min_margin=min_margin, accept_any_fitting=accept_any_fitting)


async def narrow_and_pick(
    jev: JevClient, query: str, candidates: list[str], *, instructions: str, fit_instructions: str,
    evidence_for: Callable[[list[str]], Awaitable[dict[str, str]]] | None = None,
    chunk_size: int | None = None, k: int | None = None, min_fit: float = 0.5, min_confidence: float = 0.6,
    min_margin: float = 0.15, state_extra: dict | None = None, accept_any_fitting: bool = False,
    describe: Callable[[str], str] = lambda c: c,
) -> NarrowVerdict:
    """Rounds 1+2+decide: the one function real call sites use.

    When candidates already fit in one chunk (on-screen elements, always small),
    round 1 buys nothing but a round trip and a beam_k=3 cutoff that can lose
    the right answer among otherwise-few real options -- skip straight to round
    2 with everything.

    Larger enumerations (packages, services) are decided shortlist-first on
    profiles with shortlist_first: a fuzzy-scored retrieval shortlist of
    profile.shortlist_size gets ONE choice call; the lossless chunked sweep
    runs only when that pick abstains or fails its gates. The hosted engine's
    profile keeps the 2-round chunked sweep unchanged."""
    profile = current_profile(jev.engine_name)
    chunk_size = profile.chunk_size if chunk_size is None else chunk_size
    k = BEAM_K if k is None else k
    if len(candidates) <= chunk_size:
        shortlist = candidates
    else:
        if profile.shortlist_first:
            probe = fuzzy_narrow(query, list(candidates), limit=profile.shortlist_size)
            verdict = await _pick_and_decide(
                jev, query, probe, instructions=instructions, fit_instructions=fit_instructions,
                evidence_for=evidence_for, state_extra=state_extra, describe=describe,
                enumerated=candidates, min_fit=min_fit, min_confidence=min_confidence,
                min_margin=min_margin, accept_any_fitting=accept_any_fitting,
            )
            if verdict.ok:
                return verdict
            # The retrieval shortlist missed (abstain, low confidence, nothing
            # fits) -- fall back to the lossless chunked sweep: every real
            # candidate still gets asked, ordering only, nothing dropped.
        shortlist, _ = ground_candidates(
            await semantic_shortlist(jev, query, candidates, instructions=instructions,
                                      chunk_size=chunk_size, k=k, state_extra=state_extra),
            candidates,
        )
    return await _pick_and_decide(
        jev, query, shortlist, instructions=instructions, fit_instructions=fit_instructions,
        evidence_for=evidence_for, state_extra=state_extra, describe=describe,
        enumerated=candidates, min_fit=min_fit, min_confidence=min_confidence,
        min_margin=min_margin, accept_any_fitting=accept_any_fitting,
    )
