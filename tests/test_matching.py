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


def test_accept_any_fitting_picks_the_best_fit_over_a_low_confidence_top_choice() -> None:
    """Real shape from the do-not-disturb bug: the Choice winner's own fit misses the
    floor while another real, grounded candidate clears it -- accept that one instead
    of escalating over a low-confidence pick among several plausible names."""
    verdict = decide(
        "notification", {"notification": 0.33, "settings": 0.26, "manager": 0.05}, confidence=0.18,
        fits={"notification": 0.49, "settings": 0.57, "manager": 0.57},
        enumerated=["notification", "settings", "manager"], accept_any_fitting=True,
    )
    assert verdict.ok
    assert verdict.choice == "settings"
    assert verdict.alternatives == ["manager"]


def test_accept_any_fitting_still_escalates_when_nothing_fits() -> None:
    verdict = decide(
        "gmail", {"gmail": 0.9, "maps": 0.1}, confidence=0.9,
        fits={"gmail": 0.2, "maps": 0.1}, enumerated=["gmail", "maps"], accept_any_fitting=True,
    )
    assert not verdict.ok


def test_accept_any_fitting_still_rejects_an_ungrounded_fit() -> None:
    verdict = decide(
        "invented", {"invented": 0.9}, confidence=0.9,
        fits={"invented": 0.9}, enumerated=["gmail", "maps"], accept_any_fitting=True,
    )
    assert not verdict.ok


def test_accept_any_fitting_false_by_default_keeps_existing_behavior() -> None:
    verdict = decide(
        "notification", {"notification": 0.33, "settings": 0.26}, confidence=0.18,
        fits={"notification": 0.49, "settings": 0.57}, enumerated=["notification", "settings"],
    )
    assert not verdict.ok


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


def test_extract_value_spans_stops_at_a_comma_between_multiple_values() -> None:
    goal = "address it to test@example.com, with the subject Test Chain, and the body hello"
    spans = extract_value_spans(goal)
    assert "test@example.com" in spans
    assert not any("," in s for s in spans)
    assert not any(len(s) > 60 for s in spans)


def test_extract_value_spans_keeps_a_period_inside_a_value() -> None:
    assert "test@example.com" in extract_value_spans("go to test@example.com")
    assert any(s.endswith("3.14") for s in extract_value_spans("search for pi is 3.14"))
