"""NarrowVerdict.fit_key: which fit_i Noul was asked about the EXECUTED
candidate, in shortlist order -- the join key calibration needs between a
pick decision row and the fit family asked alongside it."""

from __future__ import annotations

from jevdevice.matching import decide


def test_fit_key_for_a_normal_pick() -> None:
    fits = {"a": 0.9, "b": 0.2}
    verdict = decide("a", {"a": 0.8, "b": 0.2}, 0.9, fits, ["a", "b"])
    assert verdict.ok
    assert verdict.choice == "a"
    assert verdict.shortlist == ["a", "b"]
    assert verdict.fit_key == "fit_0"


def test_fit_key_when_a_later_shortlist_entry_is_picked() -> None:
    fits = {"a": 0.9, "b": 0.9, "c": 0.9}
    verdict = decide("c", {"a": 0.2, "b": 0.2, "c": 0.6}, 0.9, fits, ["a", "b", "c"])
    assert verdict.ok
    assert verdict.fit_key == "fit_2"


def test_fit_key_for_accept_any_fitting_when_the_winner_differs_from_the_pick() -> None:
    # the judge's Choice picked "a", but "c" fits better and clears min_fit --
    # accept_any_fitting executes "c", not the raw pick.
    fits = {"a": 0.3, "b": 0.1, "c": 0.9}
    probabilities = {"a": 0.7, "b": 0.2, "c": 0.1}
    verdict = decide("a", probabilities, 0.9, fits, ["a", "b", "c"], min_fit=0.5, accept_any_fitting=True)
    assert verdict.ok
    assert verdict.choice == "c"
    assert verdict.fit_key == "fit_2"


def test_fit_key_is_none_on_abstain() -> None:
    fits = {"a": 0.1, "b": 0.1}
    verdict = decide(None, {}, 0.0, fits, ["a", "b"])
    assert not verdict.ok
    assert verdict.choice is None
    assert verdict.fit_key is None


def test_fit_key_is_none_when_the_winner_is_not_grounded() -> None:
    fits = {"a": 0.9}
    verdict = decide("not-in-shortlist", {"a": 0.9}, 0.9, fits, ["a"])
    assert not verdict.ok
    assert verdict.fit_key is None
