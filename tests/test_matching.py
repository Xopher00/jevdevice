"""Pure functions only, no network, no device — matching.py's contract."""

from __future__ import annotations

from jevdevice.matching import (
    chunk_candidates,
    decide,
    extract_value_spans,
    ground_candidates,
)


def test_ground_candidates_splits_real_from_invented() -> None:
    grounded, ungrounded = ground_candidates(["a", "z", "b"], ["a", "b", "c"])
    assert grounded == ["a", "b"]
    assert ungrounded == ["z"]


def test_chunk_candidates_is_lossless() -> None:
    candidates = [f"pkg.{i}" for i in range(450)]
    chunks = chunk_candidates(candidates, "goal", chunk_size=200)
    assert sum(len(c) for c in chunks) == len(candidates)
    assert set().union(*chunks) == set(candidates)
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_candidates_handles_empty() -> None:
    assert chunk_candidates([], "goal") == [[]]


def test_decide_approves_grounded_confident_fitting_winner() -> None:
    verdict = decide(
        "gmail", {"gmail": 0.9, "maps": 0.1}, confidence=0.9,
        fits={"gmail": 0.95, "maps": 0.1}, enumerated=["gmail", "maps"],
    )
    assert verdict.ok
    assert verdict.choice == "gmail"


def test_decide_rejects_ungrounded_winner() -> None:
    verdict = decide(
        "invented", {"invented": 0.9}, confidence=0.9,
        fits={"invented": 0.9}, enumerated=["gmail", "maps"],
    )
    assert not verdict.ok
    assert any("grounded" in r for r in verdict.reasons)


def test_decide_flags_nothing_fits_even_with_high_confidence() -> None:
    """The case confidence_gate alone cannot express: a confident top pick
    where every real candidate is actually a bad fit."""
    verdict = decide(
        "gmail", {"gmail": 0.9, "maps": 0.1}, confidence=0.9,
        fits={"gmail": 0.2, "maps": 0.1}, enumerated=["gmail", "maps"],
    )
    assert not verdict.ok
    assert any("nothing" in r for r in verdict.reasons)


def test_decide_flags_low_confidence_even_when_fit_is_high() -> None:
    verdict = decide(
        "gmail", {"gmail": 0.51, "maps": 0.49}, confidence=0.5,
        fits={"gmail": 0.9, "maps": 0.85}, enumerated=["gmail", "maps"],
    )
    assert not verdict.ok


def test_decide_no_candidates_is_a_clean_reject() -> None:
    verdict = decide(None, {}, confidence=0.0, fits={}, enumerated=["gmail"])
    assert not verdict.ok
    assert verdict.best_fit == 0.0


def test_extract_value_spans_finds_quoted_text() -> None:
    assert "hello world" in extract_value_spans("type 'hello world' into the box")


def test_extract_value_spans_finds_trailing_preposition_clause() -> None:
    spans = extract_value_spans("search for typesafe ai")
    assert "typesafe ai" in spans


def test_extract_value_spans_over_finds_multiple_candidates() -> None:
    spans = extract_value_spans('send "hi there" to Alice')
    assert "hi there" in spans
    assert "Alice" in spans


def test_extract_value_spans_deduplicates_and_drops_empty() -> None:
    spans = extract_value_spans("open the app")
    assert "" not in spans
    assert len(spans) == len(set(spans))
