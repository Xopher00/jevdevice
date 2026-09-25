"""Proposal/result label_keys + gate_key coverage (calibration plan, wave 2D):
every proposal/result that reaches `dispatch.label_target` carries the chosen
candidate's fit_i (when a fit noul was actually asked) and the gate's own key,
and every *.fit ask is tagged with its qid so it pools onto the "fit" unit.

Fakes follow tests/test_outcome_coverage.py's shape (FakeJudge/FakeTransport);
the journal recorder here also captures `record_decision`'s `questions` (the
QuestionRef map) so a fit ask's tag can be asserted directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from typesymbolic.judge import AskResult
from typesymbolic.question import Answer

from jevdevice.actions import ui
from jevdevice.actions.app_launch import LaunchOutcome, launch_app_for_goal
from jevdevice.actions.services import (
    ToggleProposal,
    propose_dnd,
    propose_keyevent,
    propose_toggle,
    run_dumpsys_query,
)
from jevdevice.actions.ui import (
    _ELEMENT_CACHE,
    ScrollToFindOutcome,
    propose_swipe,
    propose_tap,
    propose_type,
    scroll_to_find,
)
from jevdevice.budget import current_profile
from jevdevice.execution import dispatch
from jevdevice.judge.gate import ClosedSetProposal

# --- fakes --------------------------------------------------------------------


class FakeJudge:
    """Scripted ask_all(): pops one AskResult per call, records every call."""

    def __init__(self, engine_name: str, payloads: list[dict]) -> None:
        self.name = engine_name
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    async def ask_all(self, state, questions) -> AskResult:
        self.calls.append({"state": state, "questions": questions})
        return AskResult(answers=self.payloads.pop(0))


@dataclass
class RunResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class FakeTransport:
    """Canned adb results: command -> RunResult, plus one XML dump for dump_screen."""

    def __init__(self, results: dict[str, RunResult], dump_xml: str = "") -> None:
        self._results = results
        self._dump_xml = dump_xml

    async def run(self, command: str, timeout: float = 15.0) -> RunResult:
        return self._results.get(command, RunResult())

    async def dump_hierarchy(self) -> str:
        return self._dump_xml

    async def window_size(self) -> tuple[int, int]:
        return 1080, 1920


class DecisionRecordingJournal:
    """Records every record_decision row, including `questions` (the QuestionRef
    map ask_batch builds from a tagged Question's qid/vocab_version)."""

    def __init__(self) -> None:
        self.decisions: list[dict] = []

    def record_decision(self, **row) -> None:
        self.decisions.append(row)

    def record_outcome(self, **row) -> None:
        pass

    def record_verdict(self, **row) -> None:
        pass

    def flush(self) -> None:
        pass

    def labeled_pairs(self, *args, **kwargs) -> list:
        return []


@pytest.fixture()
def journal(monkeypatch):
    from jevdevice.journal import decision_log

    recorder = DecisionRecordingJournal()
    monkeypatch.setattr(decision_log, "get_journal", lambda: recorder)
    return recorder


NONE_OF_THESE = "none_of_these"


def _choice(choice: str, probabilities: dict[str, float], confidence: float = 0.9) -> Answer:
    return Answer.from_choice(qid="", choice=choice, probabilities=probabilities, confidence=confidence)


def _noul(noul: float) -> Answer:
    return Answer.from_noul(qid="", noul=noul)


# --- fused tap: chosen index i -> label_keys=("fit_i",), gate_key=f"safe_i" ----

_TWO_BUTTON_DUMP = (
    '<hierarchy>'
    '<node text="OK" resource-id="" content-desc="" clickable="true" bounds="[0,0][100,50]"/>'
    '<node text="Cancel" resource-id="" content-desc="" clickable="true" bounds="[0,50][100,100]"/>'
    '</hierarchy>'
)


async def test_fused_tap_label_keys_and_gate_key_match_the_chosen_index(journal) -> None:
    # options = {"text='OK'": ..., "text='Cancel'": ...} -- Cancel is index 1.
    judge = FakeJudge("jev", [{
        "pick": _choice("text='Cancel'", {"text='OK'": 0.05, "text='Cancel'": 0.9}),
        "fit_0": _noul(0.2), "fit_1": _noul(0.9),
        "safe_0": _noul(0.5), "safe_1": _noul(0.9),
    }])
    transport = FakeTransport({}, dump_xml=_TWO_BUTTON_DUMP)

    proposal = await propose_tap(judge, transport, "press cancel", verbose=False)

    assert proposal.ready is not None  # safe_1=0.9 clears the 0.8 gate threshold
    assert proposal.label_keys == ("fit_1",)
    assert proposal.gate_key == "safe_1"

    target = dispatch.label_target(proposal, "tap")
    assert target.keys == ("pick", "fit_1")
    assert target.call_id == target.gate_call_id  # fused: pick + gate share one call_id
    assert target.gate_key == "safe_1"

    refs = journal.decisions[0]["questions"]
    assert refs["fit_1"].qid == "tap.fit"
    assert refs["fit_1"].group == "fit"


# --- fallback tap (narrow_and_pick + a separate gate_command ask) -------------

_ONE_BUTTON_DUMP = (
    '<hierarchy><node text="OK" resource-id="" content-desc="" clickable="true" bounds="[0,0][100,50]"/></hierarchy>'
)


async def test_fallback_tap_label_key_from_narrowing_and_separate_safe_gate(monkeypatch, journal) -> None:
    # Force the fallback branch (options > chunk_size) without needing 200+ real elements.
    tiny_chunk = replace(current_profile("jev"), chunk_size=0)
    monkeypatch.setattr(ui, "current_profile", lambda name=None: tiny_chunk)

    judge = FakeJudge("jev", [
        {"pick": _choice("text='OK'", {"text='OK'": 0.9}), "fit_0": _noul(0.9)},  # narrow_and_pick's own ask
        {"safe": _noul(0.9)},  # the separate gate_command ask
    ])
    transport = FakeTransport({}, dump_xml=_ONE_BUTTON_DUMP)

    proposal = await propose_tap(judge, transport, "press ok", verbose=False)

    assert proposal.ready is not None
    assert proposal.label_keys == ("fit_0",)
    assert proposal.gate_key == "safe"

    target = dispatch.label_target(proposal, "tap")
    assert target.keys == ("pick", "fit_0")
    assert target.call_id != target.gate_call_id  # separate asks
    assert target.gate_key == "safe"


# --- cache hit: no fit asked -> label_keys=() ----------------------------------


async def test_cache_hit_tap_has_no_label_keys_but_still_gates(journal) -> None:
    cache_key = (None, "press ok")  # foreground_package(dump_xml) is None: no package= attr
    _ELEMENT_CACHE[cache_key] = "text='OK'"
    try:
        judge = FakeJudge("jev", [{"safe": _noul(0.9)}])  # only the gate ask runs
        transport = FakeTransport({}, dump_xml=_ONE_BUTTON_DUMP)

        proposal = await propose_tap(judge, transport, "press ok", verbose=False)

        assert proposal.ready is not None
        assert proposal.label_keys == ()
        assert proposal.gate_key == "safe"
    finally:
        _ELEMENT_CACHE.pop(cache_key, None)


# --- type_text: fused field+value ask, chosen field's fit_i -------------------

_TWO_FIELD_DUMP = (
    '<hierarchy>'
    '<node text="" resource-id="field_a" content-desc="" class="android.widget.EditText" '
    'clickable="true" bounds="[0,0][100,50]"/>'
    '<node text="" resource-id="field_b" content-desc="" class="android.widget.EditText" '
    'clickable="true" bounds="[0,50][100,100]"/>'
    '</hierarchy>'
)


async def test_type_text_label_keys_are_the_chosen_field_fit_i(journal) -> None:
    judge = FakeJudge("jev", [{
        "pick": _choice("resource-id='field_b'", {"resource-id='field_a'": 0.05, "resource-id='field_b'": 0.9}),
        "fit_0": _noul(0.2), "fit_1": _noul(0.9),
        "value": _choice("hello", {"hello": 0.9}), "any_fit_value": _noul(0.9),
    }, {"safe": _noul(0.9)}])  # gate_command's own separate ask
    transport = FakeTransport({}, dump_xml=_TWO_FIELD_DUMP)

    proposal = await propose_type(judge, transport, "type 'hello' into field_b", verbose=False)

    assert proposal.ready is not None
    assert proposal.label_keys == ("fit_1",)
    assert proposal.gate_key == "safe"

    refs = journal.decisions[0]["questions"]
    assert refs["fit_1"].qid == "type_field.fit"


async def test_type_text_cache_hit_has_no_label_keys(journal) -> None:
    cache_key = (None, "type 'hello' into field_a")
    _ELEMENT_CACHE[cache_key] = "resource-id='field_a'"
    try:
        judge = FakeJudge("jev", [
            {"value": _choice("hello", {"hello": 0.9}), "any_fit": _noul(0.9)},  # _pick_value's ask
            {"safe": _noul(0.9)},  # gate_command's ask
        ])
        transport = FakeTransport({}, dump_xml=_TWO_FIELD_DUMP)

        proposal = await propose_type(judge, transport, "type 'hello' into field_a", verbose=False)

        assert proposal.ready is not None
        assert proposal.label_keys == ()
    finally:
        _ELEMENT_CACHE.pop(cache_key, None)


# --- closed-set kinds (keyevent/dnd/swipe): no fit, gate_key mirrors the ask ---


async def test_closed_set_gate_key_is_none_when_the_command_is_read_only(journal) -> None:
    # "input keyevent KEYCODE_HOME" is classified read-only -- gate_command never asks.
    judge = FakeJudge("jev", [{
        "pick": _choice("HOME", {"HOME": 0.9}), "any_fit": _noul(0.9),
    }])
    proposal = await propose_keyevent(judge, "go home", verbose=False)
    assert isinstance(proposal, ClosedSetProposal)
    assert proposal.ready is not None
    assert proposal.label_keys == ()
    assert proposal.gate_key is None
    target = dispatch.label_target(proposal, "keyevent")
    assert target.keys == ("pick",)
    assert target.gate_key is None


async def test_closed_set_gate_key_is_safe_when_a_real_gate_ask_ran(journal) -> None:
    judge = FakeJudge("jev", [
        {"pick": _choice("off", {"off": 0.9}), "any_fit": _noul(0.9)},
        {"safe": _noul(0.9)},
    ])
    proposal = await propose_dnd(judge, "turn off do not disturb", verbose=False)
    assert proposal.ready is not None
    assert proposal.label_keys == ()
    assert proposal.gate_key == "safe"


async def test_propose_swipe_closed_set_labeling(journal) -> None:
    judge = FakeJudge("jev", [
        {"pick": _choice("down", {"down": 0.9}), "any_fit": _noul(0.9)},
        {"safe": _noul(0.9)},
    ])
    transport = FakeTransport({})
    proposal = await propose_swipe(judge, transport, "scroll down", verbose=False)
    assert proposal.ready is not None
    assert proposal.label_keys == ()
    assert proposal.gate_key == "safe"


# --- toggle_service: separate gate_command ask, no fit ------------------------


async def test_toggle_service_label_keys_and_gate_key(journal) -> None:
    judge = FakeJudge("jev", [
        {
            "service": _choice("bluetooth", {"bluetooth": 0.95, "nfc": 0.03, "data": 0.02}),
            "enabled": _choice("on", {"on": 0.95, "off": 0.05}),
            "names_one": _noul(0.95),
        },
        {"safe": _noul(0.9)},
    ])
    proposal = await propose_toggle(judge, FakeTransport({}), "turn on bluetooth", verbose=False)
    assert isinstance(proposal, ToggleProposal)
    assert proposal.ready is not None
    assert proposal.label_keys == ()
    assert proposal.gate_key == "safe"
    target = dispatch.label_target(proposal, "toggle_service")
    assert target.keys == ("service",)


# --- open_app: ungated, label_keys from the executed launch pick's fit_key ----


async def test_launch_app_for_goal_label_keys_and_ungated_gate_key(journal) -> None:
    dump_xml = (
        '<hierarchy><node text="" resource-id="com.calc/id.pad" content-desc="" '
        'clickable="false" bounds="[0,0][100,100]"/></hierarchy>'
    )
    transport = FakeTransport({
        "pm list packages": RunResult(stdout="package:com.calc\npackage:com.other\n"),
        "monkey -p com.calc 1": RunResult(stdout="Events injected: 1\n"),
    }, dump_xml=dump_xml)
    judge = FakeJudge("jev", [
        {
            "pick": _choice("com.calc", {"com.calc": 0.9, "com.other": 0.05}),
            "fit_0": _noul(0.9), "fit_1": _noul(0.1),
        },
        {"satisfied": _noul(0.9)},
    ])
    outcome = await launch_app_for_goal(judge, transport, "open the calculator", verbose=False)
    assert isinstance(outcome, LaunchOutcome)
    assert outcome.launched
    assert outcome.label_keys == ("fit_0",)
    assert outcome.gate_key is None

    refs = journal.decisions[0]["questions"]
    assert refs["fit_0"].qid == "open_app.fit"

    target = dispatch.label_target(outcome, "open_app")
    assert target.keys == ("pick", "fit_0")
    assert target.gate_key is None


# --- scroll_to_find: label_keys from the final (resolving) verdict ------------


async def test_scroll_to_find_label_keys_from_the_resolving_verdict(journal) -> None:
    dump_xml = (
        '<hierarchy><node text="Battery saver" resource-id="" content-desc="" '
        'clickable="true" bounds="[0,0][100,50]"/></hierarchy>'
    )
    transport = FakeTransport({}, dump_xml=dump_xml)
    judge = FakeJudge("jev", [{
        "pick": _choice("text='Battery saver'", {"text='Battery saver'": 0.9}),
        "fit_0": _noul(0.9),
    }])
    outcome = await scroll_to_find(judge, transport, "find battery saver", verbose=False)
    assert isinstance(outcome, ScrollToFindOutcome)
    assert outcome.found == "text='Battery saver'"
    assert outcome.label_keys == ("fit_0",)
    assert outcome.gate_key is None

    refs = journal.decisions[0]["questions"]
    assert refs["fit_0"].qid == "scroll_to_find.fit"


# --- dumpsys: label_keys follow whichever ask's call_id the result carries ----


async def test_run_dumpsys_query_label_keys_prefer_the_field_ask(journal) -> None:
    transport = FakeTransport({
        "dumpsys -l": RunResult(stdout="Currently running services:\n  battery\n  window\n"),
        "dumpsys battery": RunResult(stdout="level: 88\nstatus: 2\n"),
    })
    judge = FakeJudge("jev", [
        {
            "pick": _choice("battery", {"battery": 0.9, "window": 0.05}),
            "fit_0": _noul(0.9), "fit_1": _noul(0.1),
        },
        {
            "pick": _choice("level", {"level": 0.9, "status": 0.05}),
            "fit_0": _noul(0.9), "fit_1": _noul(0.1),
        },
    ])
    outcome = await run_dumpsys_query(judge, transport, "what is the battery level?", verbose=False)
    assert outcome.service == "battery" and outcome.answer_key == "level"
    assert outcome.label_keys == ("fit_0",)  # the field ask's own fit_0, over ["level", "status"]
    assert outcome.gate_key is None

    field_refs = journal.decisions[1]["questions"]
    assert field_refs["fit_0"].qid == "dumpsys_field.fit"
    query_refs = journal.decisions[0]["questions"]
    assert query_refs["fit_0"].qid == "dumpsys_query.fit"


async def test_run_dumpsys_query_empty_parse_does_not_raise(journal) -> None:
    """Regression: field_verdict was only assigned inside `if parsed:`, so an
    empty parse raised UnboundLocalError at the final `field_verdict.ok` read."""
    transport = FakeTransport({
        "dumpsys -l": RunResult(stdout="Currently running services:\n  battery\n"),
        "dumpsys battery": RunResult(stdout="not a key value line at all\n"),
    })
    judge = FakeJudge("jev", [{
        "pick": _choice("battery", {"battery": 0.9}),
        "fit_0": _noul(0.9),
    }])
    outcome = await run_dumpsys_query(judge, transport, "what is the battery level?", verbose=False)
    assert outcome.service == "battery"
    assert outcome.parsed == {}
    assert outcome.answer_key is None
    assert outcome.label_keys == ("fit_0",)  # no field ask ran: falls back to the service pick
    assert outcome.call_id == journal.decisions[0]["call_id"]
