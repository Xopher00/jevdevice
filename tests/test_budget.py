"""Per-engine budget profiles, shortlist-first narrowing, the abstain option,
and short raw element options. Fake judges script the judge answers; nothing
touches the network or a device."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
from typesymbolic.judge import AskResult
from typesymbolic.question import Answer

from jevdevice.actions.elements import parse_actionable_elements, short_options
from jevdevice.budget import (
    JEV_PROFILE,
    LAYA_PROFILE,
    NONE_OF_THESE,
    choice_criteria,
    current_profile,
    is_abstain,
    profile_for,
)
from jevdevice.journal import decision_log
from jevdevice.judge.gate import propose_from_closed_set
from jevdevice.judge.narrowing import narrow_and_pick


class RecordingJournal:
    def __init__(self) -> None:
        self.decisions: list[dict] = []
        self.outcomes: list[dict] = []

    def record_decision(self, **row) -> None:
        self.decisions.append(row)

    def record_outcome(self, **row) -> None:
        self.outcomes.append(row)


@pytest.fixture(autouse=True)
def _no_real_journal(monkeypatch):
    monkeypatch.setattr(decision_log, "_default_journal", RecordingJournal())


class FakeJudge:
    """Scripted ask_all(): pops one AskResult per call, records every call."""

    def __init__(self, name: str, payloads: list[dict]) -> None:
        self.name = name
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    async def ask_all(self, state, questions) -> AskResult:
        self.calls.append({"state": state, "questions": questions})
        return AskResult(answers=self.payloads.pop(0))


def _choice(choice: str, probabilities: dict[str, float], confidence: float) -> Answer:
    return Answer.from_choice(qid="", choice=choice, probabilities=probabilities, confidence=confidence)


def _noul(noul: float) -> Answer:
    return Answer.from_noul(qid="", noul=noul)


def _round2_payload(choice: str, n_options: int, *, confidence: float = 0.9, fit: float = 0.9) -> dict:
    return {
        "pick": _choice(choice, {choice: 0.85, NONE_OF_THESE: 0.15}, confidence),
        **{f"fit_{i}": _noul(fit) for i in range(n_options)},
    }


def _chunk_payload(choice: str) -> dict:
    return {
        "any": _noul(0.9),
        "pick": _choice(choice, {choice: 0.8, NONE_OF_THESE: 0.2}, 0.8),
    }


# --- profiles -----------------------------------------------------------------

def test_profile_selection_keys_off_the_answering_engine() -> None:
    assert profile_for("laya") is LAYA_PROFILE
    assert profile_for("jev") is JEV_PROFILE
    # An unknown name (a test double without a name) gets the historical knobs.
    assert profile_for("bogus") is JEV_PROFILE


def test_current_profile_falls_back_to_the_process_env(monkeypatch) -> None:
    monkeypatch.delenv("JEV_ENGINE", raising=False)
    assert current_profile() is JEV_PROFILE
    monkeypatch.setenv("JEV_ENGINE", "laya")
    assert current_profile() is LAYA_PROFILE
    assert current_profile("jev") is JEV_PROFILE  # explicit engine wins over env


def test_current_profile_overlays_promoted_values(monkeypatch, tmp_path) -> None:
    from typesymbolic.calibration_store import CalibrationStore
    from typesymbolic.vocab import unit_name

    from jevdevice.calibrate import units

    calibration_store = CalibrationStore(tmp_path / "calibration")
    monkeypatch.setattr(units, "_store", calibration_store)
    calibration_store.set(
        unit_name("gate", "noul_p"), 0.93, engine="jev", default=JEV_PROFILE.gate_threshold, n=25,
    )
    calibration_store.set(
        unit_name("fit", "noul_p"), 0.71, engine="jev", default=JEV_PROFILE.min_fit, n=25,
    )
    calibration_store.set(
        unit_name("pick", "confidence"), 0.66, engine="jev", default=JEV_PROFILE.min_confidence, n=25,
    )
    profile = current_profile("jev")
    assert profile.gate_threshold == 0.93
    assert profile.min_fit == 0.71
    assert profile.min_confidence == 0.66


def test_current_profile_never_overlays_min_margin(monkeypatch, tmp_path) -> None:
    from typesymbolic.calibration_store import CalibrationStore
    from typesymbolic.vocab import unit_name

    from jevdevice.calibrate import units

    calibration_store = CalibrationStore(tmp_path / "calibration")
    monkeypatch.setattr(units, "_store", calibration_store)
    calibration_store.set(
        unit_name("pick", "margin"), 0.42, engine="jev", default=JEV_PROFILE.min_margin, n=25,
    )
    assert current_profile("jev").min_margin == JEV_PROFILE.min_margin


def test_profile_for_stays_pure_defaults(monkeypatch, tmp_path) -> None:
    from typesymbolic.calibration_store import CalibrationStore
    from typesymbolic.vocab import unit_name

    from jevdevice.calibrate import units

    calibration_store = CalibrationStore(tmp_path / "calibration")
    monkeypatch.setattr(units, "_store", calibration_store)
    calibration_store.set(
        unit_name("gate", "noul_p"), 0.93, engine="jev", default=JEV_PROFILE.gate_threshold, n=25,
    )
    assert profile_for("jev") is JEV_PROFILE  # defaults, never overlaid


def test_hosted_profile_keeps_the_historical_values() -> None:
    assert JEV_PROFILE.chunk_size == 200
    assert JEV_PROFILE.screen_limit == 150
    assert not JEV_PROFILE.shortlist_first
    assert not JEV_PROFILE.abstain_option


# --- the abstain option -------------------------------------------------------

def test_choice_criteria_offers_the_abstain_only_where_the_profile_says() -> None:
    options = {"a": None, "b": "desc"}
    assert choice_criteria(options, JEV_PROFILE) == options  # hosted engine: unchanged
    laya = choice_criteria(options, LAYA_PROFILE)
    assert laya["a"] is None and laya["b"] == "desc"
    assert laya[NONE_OF_THESE]


def test_choice_criteria_never_shadows_a_real_option_named_none_of_these() -> None:
    options = {NONE_OF_THESE: "a real option that happens to share the name"}
    assert choice_criteria(options, LAYA_PROFILE) == options


def test_is_abstain() -> None:
    assert is_abstain(NONE_OF_THESE)
    assert not is_abstain("pkg.a")
    assert not is_abstain(None)


async def test_abstain_pick_escalates_in_a_closed_set() -> None:
    judge = FakeJudge("laya", [{
        "pick": _choice(NONE_OF_THESE, {NONE_OF_THESE: 0.95}, 0.95),
        "any_fit": _noul(0.1),
    }])
    proposal = await propose_from_closed_set(
        judge, "press home", dict.fromkeys(["HOME", "BACK"]),
        options_key="key_options",
        pick_instructions="Which key?", any_fit_instructions="Does any fit?",
        pick_verb="key",
        command_for=lambda key: None, label_for=lambda key: key,
        gate_instructions="safe?", verbose=False,
    )
    assert proposal.pick is None
    assert any("abstained" in reason for reason in proposal.reasons)
    # The state still describes the real options; only the Choice gained the abstain.
    assert judge.calls[0]["state"]["key_options"] == {"HOME": None, "BACK": None}
    assert set(judge.calls[0]["questions"]["pick"].criteria) == {"HOME", "BACK", NONE_OF_THESE}


# --- shortlist-first narrowing with chunked-sweep fallback --------------------

def _candidates(n: int) -> list[str]:
    return [f"pkg.{i}" for i in range(n)]


async def test_shortlist_first_resolves_in_one_choice_call() -> None:
    judge = FakeJudge("laya", [_round2_payload("pkg.7", LAYA_PROFILE.shortlist_size)])
    verdict = await narrow_and_pick(
        judge, "open package 7", _candidates(50),
        instructions="Which package?", fit_instructions="Does {candidate} fit?",
    )
    assert verdict.ok and verdict.choice == "pkg.7"
    assert len(judge.calls) == 1  # ONE choice call for the whole high-cardinality decision
    assert len(judge.calls[0]["state"]["candidates"]) == LAYA_PROFILE.shortlist_size  # state: real options only
    probe_criteria = judge.calls[0]["questions"]["pick"].criteria
    assert len(probe_criteria) == LAYA_PROFILE.shortlist_size + 1  # option list: shortlist + the abstain option
    assert NONE_OF_THESE in probe_criteria


async def test_shortlist_miss_falls_back_to_the_lossless_chunked_sweep() -> None:
    # 50 candidates / chunk 20 -> 3 chunks. Probe abstains, the sweep covers
    # every candidate, and the final pick is a real one.
    payloads = [_round2_payload(NONE_OF_THESE, LAYA_PROFILE.shortlist_size)]
    payloads += [_chunk_payload(f"pkg.{i}") for i in (3, 23, 43)]  # one per chunk
    payloads.append(_round2_payload("pkg.43", 3))
    judge = FakeJudge("laya", payloads)
    verdict = await narrow_and_pick(
        judge, "open package 43", _candidates(50),
        instructions="Which package?", fit_instructions="Does {candidate} fit?",
    )
    assert verdict.ok and verdict.choice == "pkg.43"
    assert len(judge.calls) == 5  # probe + 3 chunks + final round 2
    chunk_states = [call["state"]["candidates"] for call in judge.calls[1:4]]
    assert all(len(chunk) <= LAYA_PROFILE.chunk_size for chunk in chunk_states)
    covered = {c for chunk in chunk_states for c in chunk}
    assert covered == set(_candidates(50))  # lossless: nothing dropped


async def test_merged_sweep_shortlist_is_capped_at_the_profile_shortlist() -> None:
    # 150 candidates -> 8 chunks x 3 beam winners = 24, which would exceed the
    # head budget in the round-2 Choice (the engine silently drops options
    # rather than error) -- the merge caps at the profile's shortlist_size.
    def chunk_with_three_winners(base: int) -> dict:
        picks = [f"pkg.{base}", f"pkg.{base + 1}", f"pkg.{base + 2}"]
        return {
            "any": _noul(0.9),
            "pick": _choice(picks[0], {picks[0]: 0.5, picks[1]: 0.3, picks[2]: 0.2}, 0.5),
        }

    payloads = [_round2_payload(NONE_OF_THESE, LAYA_PROFILE.shortlist_size)]  # probe abstains
    payloads += [chunk_with_three_winners(i * 20) for i in range(8)]  # 150 candidates -> 8 chunks
    payloads.append(_round2_payload("pkg.0", 20))
    judge = FakeJudge("laya", payloads)
    verdict = await narrow_and_pick(
        judge, "open package 0", _candidates(150),
        instructions="Which package?", fit_instructions="Does {candidate} fit?",
    )
    assert verdict.ok and verdict.choice == "pkg.0"
    round2_options = judge.calls[-1]["questions"]["pick"].criteria
    assert len(round2_options) <= LAYA_PROFILE.shortlist_size + 1  # + the abstain option


async def test_hosted_engine_keeps_the_two_round_chunked_sweep() -> None:
    # No retrieval shortlist on the jev profile: chunk asks straight away
    # (250 candidates over the hosted engine's 200-candidate chunk -> 2 chunks).
    payloads = [_chunk_payload("pkg.3"), _chunk_payload("pkg.230")]
    payloads.append(_round2_payload("pkg.230", 2))
    judge = FakeJudge("jev", payloads)
    verdict = await narrow_and_pick(
        judge, "open package 230", _candidates(250),
        instructions="Which package?", fit_instructions="Does {candidate} fit?",
    )
    assert verdict.ok and verdict.choice == "pkg.230"
    assert len(judge.calls) == 3  # 2 chunks + round 2 -- no probe ask
    assert NONE_OF_THESE not in judge.calls[0]["questions"]["pick"].criteria


async def test_abstain_pick_never_enters_the_round1_shortlist() -> None:
    # A chunk whose winner IS the abstain must contribute no candidate.
    payloads = [_round2_payload(NONE_OF_THESE, LAYA_PROFILE.shortlist_size)]  # probe abstains
    payloads += [_chunk_payload(NONE_OF_THESE), _chunk_payload("pkg.23"), _chunk_payload("pkg.43")]
    payloads.append(_round2_payload("pkg.23", 2))
    judge = FakeJudge("laya", payloads)
    verdict = await narrow_and_pick(
        judge, "open package 23", _candidates(50),
        instructions="Which package?", fit_instructions="Does {candidate} fit?",
    )
    assert verdict.ok and verdict.choice == "pkg.23"
    assert NONE_OF_THESE not in verdict.shortlist


# --- short raw element options -------------------------------------------------

DUMP_TWO_SENDS = (
    '<hierarchy>'
    '<node text="Send" resource-id="com.x:id/one" content-desc="" clickable="true" bounds="[0,0][100,50]"/>'
    '<node text="Send" resource-id="com.x:id/two" content-desc="" clickable="true" bounds="[0,50][100,100]"/>'
    '</hierarchy>'
)


def test_short_options_prefers_text_then_deduplicates_collisions() -> None:
    elements = parse_actionable_elements(DUMP_TWO_SENDS)
    options = short_options(elements)
    assert list(options) == ["Send", "Send #2"]
    assert options["Send"].bounds == "[0,0][100,50]"
    assert options["Send #2"].bounds == "[0,50][100,100]"


def test_describe_screen_limit_comes_from_the_profile(monkeypatch) -> None:
    from jevdevice.actions.elements import describe_screen
    nodes = "".join(
        f'<node text="item{i}" resource-id="" content-desc="" clickable="true" bounds="[0,{i}][100,{i + 10}]"/>'
        for i in range(20)
    )
    dump_xml = f"<hierarchy>{nodes}</hierarchy>"
    monkeypatch.delenv("JEV_ENGINE", raising=False)
    assert len(describe_screen(dump_xml)) == 20  # hosted profile: 150, no cut
    monkeypatch.setenv("JEV_ENGINE", "laya")
    assert len(describe_screen(dump_xml)) == LAYA_PROFILE.screen_limit
    assert len(describe_screen(dump_xml, limit=5)) == 5  # explicit limit still wins


def test_short_label_prefers_text_over_resource_id() -> None:
    dump_xml = (
        '<hierarchy>'
        '<node text="" resource-id="com.x:id/search_box" content-desc="Search" clickable="true" bounds="[0,0][100,50]"/>'
        '</hierarchy>'
    )
    options = short_options(parse_actionable_elements(dump_xml))
    assert list(options) == ["Search"]

# --- phase 4 calibration knobs ------------------------------------------------

def test_calibration_knobs_jev_keeps_the_historical_values() -> None:
    assert JEV_PROFILE.gate_threshold == 0.8
    assert JEV_PROFILE.min_confidence == 0.6
    assert JEV_PROFILE.min_margin == 0.15
    assert JEV_PROFILE.min_fit == 0.5
    assert JEV_PROFILE.noul_floor == 0.5
    assert JEV_PROFILE.temp_choice is None  # not fitted


def test_calibration_knobs_carry_labeled_sample_provenance() -> None:
    # Every fitted temperature records how many labels it rests on; an
    # unfitted type is None and absent from the provenance map.
    for profile in (JEV_PROFILE, LAYA_PROFILE):
        for temp_name in ("temp_choice", "temp_noul", "temp_score"):
            temp = getattr(profile, temp_name)
            if temp is None:
                continue
            assert profile.calibration_n.get(temp_name.removeprefix("temp_"), 0) > 0


async def test_narrow_and_pick_thresholds_come_from_the_answering_engine_profile() -> None:
    laya_like = replace(JEV_PROFILE, engine="laya", min_fit=0.2, min_confidence=0.3, min_margin=0.05, shortlist_first=False)
    with patch("jevdevice.judge.narrowing.current_profile", return_value=laya_like):
        judge = FakeJudge("laya", [_round2_payload("gm", 1, confidence=0.35, fit=0.25)])
        verdict = await narrow_and_pick(judge, "open gmail", ["gm"],
                                        instructions="Which package best satisfies the goal?",
                                        fit_instructions="Is {candidate} the app the goal asks to open?")
    assert verdict.ok  # 0.35 >= 0.3 confidence, fit 0.25 >= 0.2: profile knobs applied
