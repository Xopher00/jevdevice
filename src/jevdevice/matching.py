"""Generic candidate narrowing and decision-confidence gating. Pure, no
network: anything that asks Jev a question lives in narrowing.py instead."""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .budget import JEV_PROFILE

VALUE_PREPOSITIONS = ("for", "to", "into", "with", "and")


def extract_value_spans(goal: str) -> list[str]:
    """Regex over-finds candidate text values in a goal string; Jev picks which
    one (if any) is the value to type, and code copies it verbatim -- it never
    generates the text itself (docs.typesafe.ai/cookbooks/
    pre_parsed_value_extraction_cookbook.md). Quoted spans, plus the trailing
    clause after an ordinary English preposition -- grammatical function words,
    not per-goal templates -- so this generalizes across any goal phrasing.
    A comma ends a clause early: a goal naming several values, "to X, with Y, and Z",
    otherwise swallows all three into one span -- a period doesn't, since one can be part
    of the value itself (e.g. an email)."""
    spans: list[str] = []
    for match in re.finditer(r"['\"]([^'\"]+)['\"]", goal):
        spans.append(match.group(1))
    for prep in VALUE_PREPOSITIONS:
        match = re.search(rf"\b{prep}\s+(.+?)(?:,|[.!?]?$)", goal, re.IGNORECASE)
        if match:
            spans.append(match.group(1).strip())
    seen: set[str] = set()
    return [s for s in spans if s and not (s in seen or seen.add(s))]


def fuzzy_narrow(goal: str, candidates: list[str], limit: int = 20) -> list[str]:
    """Score each candidate by its best-matching dot/underscore-separated segment,
    not the whole string, so e.g. "calculator" matches ...app.popupcalculator."""
    goal_tokens = [t for t in goal.casefold().split() if len(t) > 3]

    def score(candidate: str) -> float:
        segments = candidate.casefold().replace("_", ".").split(".")
        return max(
            (difflib.SequenceMatcher(None, token, segment).ratio() for token in goal_tokens for segment in segments),
            default=0.0,
        )

    scored = sorted(((score(c), c) for c in candidates), reverse=True)
    return [c for _, c in scored[:limit]]


def margin(probabilities: dict[str, float]) -> float:
    """Gap between the top two picks (or the sole pick's own probability)."""
    ranked = sorted(probabilities.values(), reverse=True)
    return ranked[0] - ranked[1] if len(ranked) >= 2 else ranked[0]


def confidence_gate(probabilities: dict[str, float], confidence: float, min_confidence: float = 0.6, min_margin: float = 0.15) -> tuple[bool, str]:
    """No action below threshold — escalate instead of acting on a close or unsure pick."""
    gap = margin(probabilities)
    if confidence < min_confidence:
        return False, f"confidence {confidence:.2f} below {min_confidence}"
    if gap < min_margin:
        return False, f"margin {gap:.2f} below {min_margin} (top two picks too close)"
    return True, "ok"


def ground_candidates(proposed: Iterable[str], enumerated: Sequence[str]) -> tuple[list[str], list[str]]:
    """Zero-cost check, before any model judgment: is each proposed value
    literally present in the real enumeration? Never raises — grounding
    failure is a signal to route on, not a bug to crash over."""
    real = set(enumerated)
    grounded, ungrounded = [], []
    for value in proposed:
        (grounded if value in real else ungrounded).append(value)
    return grounded, ungrounded


def chunk_candidates(candidates: Sequence[str], query: str, chunk_size: int | None = None) -> list[list[str]]:
    """Split into <=chunk_size groups, ordered by fuzzy_narrow so the most
    plausible chunk comes first, but every candidate lands in exactly one
    chunk — ordering only, nothing is ever dropped. The bound is a named knob:
    the calling engine's profile chunk_size, never a bare int at a call site."""
    chunk_size = JEV_PROFILE.chunk_size if chunk_size is None else chunk_size
    ordered = fuzzy_narrow(query, list(candidates), limit=len(candidates))
    return [ordered[i:i + chunk_size] for i in range(0, len(ordered), chunk_size)] or [[]]


@dataclass(frozen=True)
class NarrowVerdict:
    choice: str | None
    confidence: float               # Choice confidence on the final shortlist
    fit: float                      # the winner's own "does this really fit" Noul
    best_fit: float                 # max fit Noul across the whole shortlist
    shortlist: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # every signal that fired
    alternatives: list[str] = field(default_factory=list)  # other real candidates that also cleared min_fit
    # Journal linkage: the round-2 ground ask's call_id, so the downstream
    # outcome row (open_app/dumpsys/scroll_to_find) joins the decision row that
    # picked the executed candidate. None on paths that never asked.
    call_id: str | None = None

    @property
    def ok(self) -> bool:
        return not self.reasons


def decide(
    choice: str | None, probabilities: dict[str, float], confidence: float,
    fits: dict[str, float], enumerated: Sequence[str],
    *, min_fit: float = 0.5, min_confidence: float = 0.6, min_margin: float = 0.15,
    accept_any_fitting: bool = False,
) -> NarrowVerdict:
    """MAX-aggregated verdict (sde_cascade: one red flag escalates, never
    averaged into silence). `fits` gives an independent, unnormalized signal
    per shortlist entry, which is what lets "none of these actually fit" be
    representable at all — Choice probabilities always sum to 1, so a
    confidence/margin check alone can only ever say "the top pick is unsure",
    never "everything here is bad".

    `accept_any_fitting`: for read-only queries where several real candidates can be
    independently correct -- skips the confidence/margin gate when any candidate clears
    `min_fit`, which stays the only real safety net either way."""
    shortlist = list(fits)
    best_fit = max(fits.values(), default=0.0)
    enumerated_set = set(enumerated)
    fitting = [c for c in shortlist if fits[c] >= min_fit and c in enumerated_set]

    if accept_any_fitting and fitting:
        winner = max(fitting, key=lambda c: (fits[c], probabilities.get(c, 0.0)))
        return NarrowVerdict(winner, confidence, fits[winner], best_fit, shortlist, [], [c for c in fitting if c != winner])

    winner_fit = fits.get(choice, 0.0) if choice is not None else 0.0
    reasons = []
    if choice is None or choice not in enumerated_set:
        reasons.append("winner not grounded in the real enumeration")
    if best_fit < min_fit:
        reasons.append(f"nothing in the shortlist fits (best {best_fit:.2f} below {min_fit})")
    elif winner_fit < min_fit:
        reasons.append(f"ranked winner isn't the one that fits (fit {winner_fit:.2f} below {min_fit})")
    if probabilities:
        ok, reason = confidence_gate(probabilities, confidence, min_confidence, min_margin)
        if not ok:
            reasons.append(reason)
    else:
        reasons.append("no candidates to choose among")

    return NarrowVerdict(choice, confidence, winner_fit, best_fit, shortlist, reasons)
