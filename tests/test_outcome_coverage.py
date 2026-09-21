"""Ungated-kind outcome-row coverage: open_app,
dumpsys, scroll_to_find and screenshot emit outcome rows joined by call_id,
the same journal linkage the gated kinds already have. All four run through
dispatch.run_kind, the one shared execution path.

Nothing here touches a device or the network: handlers run against fakes,
mcp_server is imported with placeholder env vars (same as test_decision_log).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("TYPESAFE_AI_API", "placeholder-for-import")
os.environ.setdefault("ANDROID_SERIAL", "placeholder-for-import")

import pytest

from jevdevice.actions.app_launch import LaunchOutcome, launch_app_for_goal
from jevdevice.actions.services import DumpsysOutcome, run_dumpsys_query
from jevdevice.actions.ui import ScrollToFindOutcome, scroll_to_find
from jevdevice.jev import ChoiceAnswer, NoulAnswer
from jevdevice.judge.narrowing import narrow_and_pick

# --- fakes --------------------------------------------------------------------

class FakeJudge:
    """Scripted ask(): pops one answers payload per call, records every call."""

    def __init__(self, engine_name: str, payloads: list[dict]) -> None:
        self.engine_name = engine_name
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    async def ask(self, state, questions, **kw):
        self.calls.append({"state": state, "questions": questions, **kw})
        return self.payloads.pop(0)


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


# --- narrowing verdicts carry the ground ask's call_id --------------------------

async def test_narrow_and_pick_attaches_the_ground_ask_call_id() -> None:
    from jevdevice.budget import NONE_OF_THESE

    judge = FakeJudge("jev", [{
        "pick": ChoiceAnswer(choice="pkg.7", probabilities={"pkg.7": 0.9, NONE_OF_THESE: 0.1}, confidence=0.9),
        "fit_0": NoulAnswer(noul=0.9),
    }])
    verdict = await narrow_and_pick(
        judge, "open package 7", ["pkg.7"],
        instructions="Which package?", fit_instructions="Does {candidate} fit?",
    )
    assert verdict.ok and verdict.choice == "pkg.7"
    # The call_id passed to ask() is the one the verdict carries out: the outcome
    # row joins the exact decision row that picked what later ran.
    assert verdict.call_id is not None
    assert judge.calls[0]["call_id"] == verdict.call_id
    assert judge.calls[0]["phase"] == "ground"


async def test_narrow_and_pick_abstain_verdict_carries_the_call_id_too() -> None:
    from jevdevice.budget import NONE_OF_THESE

    judge = FakeJudge("jev", [{
        "pick": ChoiceAnswer(choice=NONE_OF_THESE, probabilities={NONE_OF_THESE: 1.0}, confidence=0.9),
        "fit_0": NoulAnswer(noul=0.1),
    }])
    verdict = await narrow_and_pick(
        judge, "open package 7", ["pkg.7"],
        instructions="Which package?", fit_instructions="Does {candidate} fit?",
    )
    assert not verdict.ok
    assert verdict.call_id == judge.calls[0]["call_id"]


# --- dumpsys / open_app / scroll_to_find thread it end to end -------------------

async def test_run_dumpsys_query_threads_the_pick_call_id() -> None:
    from jevdevice.budget import NONE_OF_THESE

    transport = FakeTransport({
        "dumpsys -l": RunResult(stdout="Currently running services:\n  battery\n  window\n"),
        "dumpsys battery": RunResult(stdout="level: 88\nstatus: 2\n"),
    })
    judge = FakeJudge("jev", [
        {  # service pick (round 2 over 2 services)
            "pick": ChoiceAnswer(choice="battery", probabilities={"battery": 0.9, "window": 0.05, NONE_OF_THESE: 0.05}, confidence=0.9),
            "fit_0": NoulAnswer(noul=0.9),
            "fit_1": NoulAnswer(noul=0.1),
        },
        {  # answer-field pick over the parsed keys
            "pick": ChoiceAnswer(choice="level", probabilities={"level": 0.9, "status": 0.05, NONE_OF_THESE: 0.05}, confidence=0.9),
            "fit_0": NoulAnswer(noul=0.9),
            "fit_1": NoulAnswer(noul=0.1),
        },
    ])
    outcome = await run_dumpsys_query(judge, transport, "what is the battery level?", verbose=False)
    assert outcome.service == "battery" and outcome.answer_key == "level"
    # The answer-field pick's ask is the last decision before the outcome; its
    # call_id is what the outcome row joins.
    assert outcome.call_id == judge.calls[1]["call_id"]


async def test_launch_app_for_goal_threads_the_pick_call_id() -> None:
    from jevdevice.budget import NONE_OF_THESE

    dump_xml = (
        '<hierarchy><node text="" resource-id="com.calc/id.pad" content-desc="" '
        'clickable="false" bounds="[0,0][100,100]"/></hierarchy>'
    )
    transport = FakeTransport({
        "pm list packages": RunResult(stdout="package:com.calc\npackage:com.other\n"),
        "monkey -p com.calc 1": RunResult(stdout="Events injected: 1\n"),
    }, dump_xml=dump_xml)
    judge = FakeJudge("jev", [
        {  # round 2 package pick
            "pick": ChoiceAnswer(choice="com.calc", probabilities={"com.calc": 0.9, "com.other": 0.05, NONE_OF_THESE: 0.05}, confidence=0.9),
            "fit_0": NoulAnswer(noul=0.9),
            "fit_1": NoulAnswer(noul=0.1),
        },
        {  # launch verify
            "satisfied": NoulAnswer(noul=0.9),
        },
    ])
    outcome = await launch_app_for_goal(judge, transport, "open the calculator", verbose=False)
    assert outcome.launched and outcome.package == "com.calc"
    assert outcome.call_id == judge.calls[0]["call_id"]


async def test_scroll_to_find_success_carries_call_id_and_no_executed() -> None:
    from jevdevice.budget import NONE_OF_THESE

    dump_xml = (
        '<hierarchy><node text="Battery saver" resource-id="" content-desc="" '
        'clickable="true" bounds="[0,0][100,50]"/></hierarchy>'
    )
    transport = FakeTransport({}, dump_xml=dump_xml)
    judge = FakeJudge("jev", [
        {  # element pick (round 2, single candidate)
            "pick": ChoiceAnswer(choice="text='Battery saver'", probabilities={"text='Battery saver'": 0.9, NONE_OF_THESE: 0.1}, confidence=0.9),
            "fit_0": NoulAnswer(noul=0.9),
        },
    ])
    outcome = await scroll_to_find(judge, transport, "find battery saver", verbose=False)
    assert outcome.found == "text='Battery saver'" and outcome.attempts == 1
    assert outcome.call_id == judge.calls[0]["call_id"]
    assert outcome.executed == ()  # found on the first screen check: nothing swiped


# --- mcp_server's ungated handlers emit outcome rows -----------------------------

class RecordingJournal:
    def __init__(self) -> None:
        self.outcomes: list[dict] = []

    def record_outcome(self, **row) -> None:
        self.outcomes.append(row)


@pytest.fixture()
def outcome_journal(monkeypatch):
    from jevdevice.journal import decision_log

    recorder = RecordingJournal()
    monkeypatch.setattr(decision_log, "get_journal", lambda: recorder)
    # no foreground dumps in these tests: the fake transports below carry no
    # device, and the rows under test don't assert graph_edge anyway
    monkeypatch.setenv("JEV_GRAPH_EDGE", "0")
    return recorder


async def test_ungated_kinds_emit_outcome_rows(monkeypatch, outcome_journal):
    from jevdevice import mcp_server
    from jevdevice.execution import dispatch
    from jevdevice.journal.decision_log import VERIFIED

    async def fake_open(jev, transport, goal, *, verbose):
        return LaunchOutcome("com.calc", 0.9, True, 0.9, 1, "com.calc", call_id="cid-open")

    async def fake_dumpsys(jev, transport, goal, *, verbose):
        return DumpsysOutcome("battery", 0.9, 0.9, parsed={"level": "88"}, answer_key="level", call_id="cid-dump")

    async def fake_scroll(jev, transport, goal, *, direction="down", max_attempts=8, verbose=True):
        return ScrollToFindOutcome("text='Battery saver'", 2, call_id="cid-scroll",
                                   executed=("input swipe a", "input swipe b"))

    monkeypatch.setattr(dispatch, "launch_app_for_goal", fake_open)
    monkeypatch.setattr(dispatch, "run_dumpsys_query", fake_dumpsys)
    monkeypatch.setattr(dispatch, "scroll_to_find", fake_scroll)

    jev, transport = mcp_server.jev, mcp_server.transport
    open_response = await dispatch.run_kind(jev, transport, "open_app", "open the calculator")
    dump_response = await dispatch.run_kind(jev, transport, "dumpsys", "what is the battery level?")
    scroll_response = await dispatch.run_kind(jev, transport, "scroll_to_find", "find battery saver")
    screenshot_response = await dispatch.run_kind(jev, transport, "screenshot", "take a screenshot")

    rows = outcome_journal.outcomes
    assert [r["call_id"] for r in rows] == ["cid-open", "cid-dump", "cid-scroll", None]
    assert rows[0]["executed_command"] == "monkey -p com.calc 1"
    assert rows[0]["verification"] == VERIFIED
    assert rows[1]["executed_command"] == "dumpsys battery"
    assert rows[1]["verification"] == VERIFIED
    assert rows[2]["executed_command"] == "input swipe b"  # the last thing that ran
    assert rows[2]["executed"] == ["input swipe a", "input swipe b"]
    assert rows[2]["verification"] == VERIFIED
    assert rows[3]["executed_command"] == "screencap -p"
    assert all(r["verification"] == VERIFIED for r in rows)
    # The tool-visible responses are unchanged in shape (plus scroll's executed list).
    assert open_response["status"] == "ok"
    assert dump_response["status"] == "ok"
    assert scroll_response["status"] == "ok"
    assert screenshot_response == {"status": "ok"}


async def test_ungated_escalations_emit_their_outcome_row_too(monkeypatch, outcome_journal):
    from jevdevice import mcp_server
    from jevdevice.execution import dispatch
    from jevdevice.journal.decision_log import ESCALATED

    async def fake_open(jev, transport, goal, *, verbose):
        return LaunchOutcome(None, 0.2, False, 0.0, 0, "", ("judge abstained",), call_id="cid-esc")

    monkeypatch.setattr(dispatch, "launch_app_for_goal", fake_open)
    response = await dispatch.run_kind(mcp_server.jev, mcp_server.transport, "open_app", "open something unnameable")
    assert response["status"] == "escalated"
    row = outcome_journal.outcomes[0]
    assert row["call_id"] == "cid-esc"
    assert row["executed_command"] is None  # nothing ran
    assert row["verification"] == ESCALATED


# --- needs_approval hooks: sync and async alike --------------------------------
# The MCP server parks a pending action from a SYNC hook; run_kind's contract
# (Callable[..., Awaitable[dict] | dict]) admits both, so a needs_approval
# verdict must surface a plain-dict hook's return without awaiting it.

_TOGGLE_FILL = {
    # toggle fill: confident service + direction, goal names exactly one radio
    "service": ChoiceAnswer(choice="bluetooth", probabilities={"bluetooth": 0.95, "nfc": 0.03, "data": 0.02}, confidence=0.95),
    "enabled": ChoiceAnswer(choice="on", probabilities={"on": 0.95, "off": 0.05}, confidence=0.95),
    "names_one": NoulAnswer(noul=0.95),
}
_TOGGLE_GATE_UNCERTAIN = {"safe": NoulAnswer(noul=0.5)}  # below the profile floor -> NEEDS_APPROVAL


async def test_run_kind_sync_on_pending_hook_round_trips(monkeypatch, outcome_journal) -> None:
    from jevdevice.execution import dispatch

    judge = FakeJudge("jev", [dict(_TOGGLE_FILL), dict(_TOGGLE_GATE_UNCERTAIN)])
    seen: dict = {}

    def sync_hook(goal, kind, resume_arg, confidence, pending, verify):
        seen.update(goal=goal, kind=kind, resume_arg=resume_arg, confidence=confidence,
                    command=pending.command.command, verify=verify)
        return {"status": "needs_approval", "thread_id": "t1"}

    response = await dispatch.run_kind(
        judge, FakeTransport({}), "toggle_service", "turn on bluetooth", on_pending=sync_hook,
    )
    # the sync hook's plain dict must come back verbatim, un-awaited
    assert response == {"status": "needs_approval", "thread_id": "t1"}
    assert seen["goal"] == "turn on bluetooth"
    assert seen["kind"] == "toggle_service"
    assert seen["resume_arg"] == "bluetooth"
    assert seen["command"] == "svc bluetooth enable"
    assert seen["verify"] is True
    # nothing executed: the empty canned transport has no result for any command
    row = outcome_journal.outcomes[0]
    assert row["status"] == "needs_approval"
    assert row["executed_command"] is None


async def test_run_kind_async_on_pending_hook_still_round_trips(monkeypatch, outcome_journal) -> None:
    from jevdevice.execution import dispatch

    judge = FakeJudge("jev", [dict(_TOGGLE_FILL), dict(_TOGGLE_GATE_UNCERTAIN)])

    async def async_hook(goal, kind, resume_arg, confidence, pending, verify):
        return {"status": "needs_approval", "thread_id": "t2"}

    response = await dispatch.run_kind(
        judge, FakeTransport({}), "toggle_service", "turn on bluetooth", on_pending=async_hook,
    )
    assert response == {"status": "needs_approval", "thread_id": "t2"}
    assert outcome_journal.outcomes[0]["status"] == "needs_approval"
