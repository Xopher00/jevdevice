"""Generic candidate narrowing and decision-confidence gating.

Both live-verified this session: fuzzy_narrow correctly surfaced the real
calculator/clock/youtube packages and the real bluetooth/wifi/nfc dumpsys
services from real enumerations; confidence_gate correctly escalated the live
apex-vs-youtube near-tie (40%/38%) instead of acting on it.
"""

from __future__ import annotations

import difflib


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


def confidence_gate(probabilities: dict[str, float], confidence: float, min_confidence: float = 0.6, min_margin: float = 0.15) -> tuple[bool, str]:
    """No action below threshold — escalate instead of acting on a close or unsure pick."""
    ranked = sorted(probabilities.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) >= 2 else ranked[0]
    if confidence < min_confidence:
        return False, f"confidence {confidence:.2f} below {min_confidence}"
    if margin < min_margin:
        return False, f"margin {margin:.2f} below {min_margin} (top two picks too close)"
    return True, "ok"
