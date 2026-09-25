"""Real post-action check for keyevent/swipe/set_dnd (dispatch.py's KIND_TABLE):
exit code 0 alone must never read as "ok" -- the same screen re-check tap/type
already run (actions/ui._verify_after_action) has to confirm the goal too.
Also covers run_kind's tier/recipe_id -> on_pending passthrough and the MCP
approve path recording kind/tier/recipe_id on the outcome row.

Nothing here touches a device or the network: fakes only, mcp_server is
imported with placeholder env vars (same approach as test_outcome_coverage.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("TYPESAFE_AI_API", "placeholder-for-import")
os.environ.setdefault("ANDROID_SERIAL", "placeholder-for-import")

import pytest
from typesymbolic.judge import AskResult
from typesymbolic.question import Answer

from jevdevice.budget import NONE_OF_THESE
from jevdevice.execution import dispatch
from jevdevice.journal.outcomes import LabelTarget
from jevdevice.judge.gate import CommandVariant, GateResult, GateVerdict, Pending

# --- fakes (mirrors test_outcome_coverage.py) ----------------------------------


class FakeJudge:
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
    def __init__(self, results: dict[str, RunResult], dump_xml: str = "<hierarchy></hierarchy>") -> None:
        self._results = results
        self._dump_xml = dump_xml

    async def run(self, command: str, timeout: float = 15.0) -> RunResult:
        return self._results.get(command, RunResult())

    async def dump_hierarchy(self) -> str:
        return self._dump_xml


class RecordingJournal:
    def __init__(self) -> None:
        self.outcomes: list[dict] = []
        self.verdicts: list[dict] = []

    def record_decision(self, **row) -> None:
        pass

    def record_outcome(self, *, key=None, outcome=None, extra=None, **row) -> None:
        row = {**row, "outcome": outcome, "key": key if key is not None else getattr(outcome, "key", None)}
        self.outcomes.append({**row, **(extra or {})})

    def record_verdict(self, **row) -> None:
        self.verdicts.append(row)

    def flush(self) -> None:
        pass

    def labeled_pairs(self, *args, **kwargs) -> list:
        return []


@pytest.fixture()
def outcome_journal(monkeypatch):
    from jevdevice.journal import decision_log

    recorder = RecordingJournal()
    monkeypatch.setattr(decision_log, "get_journal", lambda: recorder)
    monkeypatch.setenv("JEV_GRAPH_EDGE", "0")
    return recorder


# KEYCODE_HOME is the one read-only key (judge/gate.py's _READ_ONLY_ARGV), so
# picking it skips a second gate ask -- only pick+any_fit, then the verify ask.
_PICK_HOME = {
    "pick": Answer.from_choice(qid="", choice="HOME", probabilities={"HOME": 0.95, NONE_OF_THESE: 0.05}, confidence=0.95),
    "any_fit": Answer.from_noul(qid="", noul=0.95),
}


def _satisfied(noul: float) -> dict:
    return {"satisfied": Answer.from_noul(qid="", noul=noul)}


# --- exit code + satisfied together decide status -------------------------------


async def test_keyevent_ok_on_exit_zero_and_high_satisfied(outcome_journal) -> None:
    judge = FakeJudge("jev", [dict(_PICK_HOME), _satisfied(0.9)])
    transport = FakeTransport({"input keyevent KEYCODE_HOME": RunResult(exit_code=0)})

    response = await dispatch.run_kind(judge, transport, "keyevent", "go home")

    assert response == {"status": "ok", "exit_code": 0, "satisfied": 0.9}
    assert outcome_journal.verdicts[-1]["verdict"].status == "verified"


async def test_keyevent_unverified_on_low_satisfied_despite_exit_zero(outcome_journal) -> None:
    # _verify_after_action retries across all 3 delay slots while satisfied stays low.
    judge = FakeJudge("jev", [dict(_PICK_HOME), _satisfied(0.2), _satisfied(0.2), _satisfied(0.2)])
    transport = FakeTransport({"input keyevent KEYCODE_HOME": RunResult(exit_code=0)})

    response = await dispatch.run_kind(judge, transport, "keyevent", "go home")

    assert response == {"status": "unverified", "exit_code": 0, "satisfied": 0.2}
    # A low satisfied noul never reads as verified -- no label for recipes/calibration.
    assert outcome_journal.verdicts[-1]["verdict"].status != "verified"


async def test_keyevent_failed_on_nonzero_exit(outcome_journal) -> None:
    judge = FakeJudge("jev", [dict(_PICK_HOME), _satisfied(0.9)])
    transport = FakeTransport({"input keyevent KEYCODE_HOME": RunResult(exit_code=1)})

    response = await dispatch.run_kind(judge, transport, "keyevent", "go home")

    assert response == {"status": "unverified", "exit_code": 1, "satisfied": 0.9}
    assert outcome_journal.verdicts[-1]["verdict"].status == "failed"


async def test_keyevent_verify_false_never_ok_and_skips_the_ask(outcome_journal) -> None:
    judge = FakeJudge("jev", [dict(_PICK_HOME)])
    transport = FakeTransport({"input keyevent KEYCODE_HOME": RunResult(exit_code=0)})

    response = await dispatch.run_kind(judge, transport, "keyevent", "go home", verify=False)

    assert response == {"status": "unverified", "exit_code": 0, "satisfied": None}
    assert len(judge.calls) == 1  # no verify ask ran


# --- run_kind passes tier/recipe_id to on_pending, alongside label= -----------

_TOGGLE_FILL = {
    "service": Answer.from_choice(qid="", choice="bluetooth", probabilities={"bluetooth": 0.95, "nfc": 0.03, "data": 0.02}, confidence=0.95),
    "enabled": Answer.from_choice(qid="", choice="on", probabilities={"on": 0.95, "off": 0.05}, confidence=0.95),
    "names_one": Answer.from_noul(qid="", noul=0.95),
}
_TOGGLE_GATE_UNCERTAIN = {"safe": Answer.from_noul(qid="", noul=0.5)}  # below the profile floor -> NEEDS_APPROVAL


async def test_run_kind_passes_tier_and_recipe_id_to_on_pending(outcome_journal) -> None:
    judge = FakeJudge("jev", [dict(_TOGGLE_FILL), dict(_TOGGLE_GATE_UNCERTAIN)])
    seen: dict = {}

    def hook(goal, kind, resume_arg, confidence, pending, verify, **kwargs):
        seen.update(kwargs)
        return {"status": "needs_approval", "thread_id": "t1"}

    await dispatch.run_kind(
        judge, FakeTransport({}), "toggle_service", "turn on bluetooth",
        on_pending=hook, tier=2, recipe_id="rid-9",
    )

    assert seen["tier"] == 2
    assert seen["recipe_id"] == "rid-9"
    assert "label" in seen


# --- mcp_server.device_approve records kind, tier, recipe_id -------------------


@pytest.fixture()
def mcp_env(monkeypatch, outcome_journal):
    from jevdevice import mcp_server

    judge = FakeJudge("jev", [_satisfied(0.9)])
    transport = FakeTransport({"input keyevent KEYCODE_HOME": RunResult(exit_code=0)})
    monkeypatch.setattr(mcp_server, "jev", judge)
    monkeypatch.setattr(mcp_server, "transport", transport)
    return mcp_server


def _pending_keyevent_action(mcp_server, *, label=None, call_id="cid-gate", tier=3, recipe_id="rid-7"):
    gate_result = GateResult(verdict=GateVerdict.NEEDS_APPROVAL, reason="jev_uncertain", confidence=0.5, call_id=call_id)
    pending = Pending(command=CommandVariant("input keyevent KEYCODE_HOME", "press HOME per the goal"),
                      chosen_label="press HOME", gate_result=gate_result)
    return mcp_server.PendingAction(
        goal="go home", kind="keyevent", resume_arg=None, confidence=0.9, pending=pending,
        verify=True, call_id=call_id, label=label, tier=tier, recipe_id=recipe_id,
    )


async def test_device_approve_fallback_path_records_kind_and_tier(mcp_env) -> None:
    action_id = "thread-1"
    mcp_env._PENDING[action_id] = _pending_keyevent_action(mcp_env, label=None)

    response = await mcp_env.device_approve(thread_id=action_id, decision="approve")

    assert response[0]["status"] == "ok"
    row = mcp_env._PENDING  # already popped
    assert action_id not in row
    outcome_row = None
    from jevdevice.journal import decision_log
    recorder = decision_log.get_journal()
    outcome_row = recorder.outcomes[-1]
    assert outcome_row["kind"] == "keyevent"
    assert outcome_row["tier"] == 3
    assert outcome_row["recipe_id"] == "rid-7"


async def test_device_approve_label_path_records_kind_and_tier(mcp_env) -> None:
    action_id = "thread-2"
    label = LabelTarget(call_id="cid-label", keys=("pick",))
    mcp_env._PENDING[action_id] = _pending_keyevent_action(mcp_env, label=label, call_id="cid-label", tier=1, recipe_id="rid-2")

    await mcp_env.device_approve(thread_id=action_id, decision="approve")

    from jevdevice.journal import decision_log
    recorder = decision_log.get_journal()
    outcome_row = recorder.outcomes[-1]
    assert outcome_row["kind"] == "keyevent"
    assert outcome_row["tier"] == 1
    assert outcome_row["recipe_id"] == "rid-2"


async def test_device_approve_deny_path_records_kind(mcp_env) -> None:
    action_id = "thread-3"
    mcp_env._PENDING[action_id] = _pending_keyevent_action(mcp_env, label=None)

    response = await mcp_env.device_approve(thread_id=action_id, decision="deny")

    assert response[0]["status"] == "not_executed"
    from jevdevice.journal import decision_log
    recorder = decision_log.get_journal()
    outcome_row = recorder.outcomes[-1]
    assert outcome_row["kind"] == "keyevent"
