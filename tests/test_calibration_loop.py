"""End-to-end proof of the live calibration loop (plan part E): real
`dispatch.run_kind`/`_run_ungated`/approval-hook paths, scripted judges and
fake devices, a REAL `typesymbolic.Journal` under the autouse isolated
`JEV_JOURNAL_DIR` (see conftest.py) so `labeled_pairs` is core's own
`LabelIndex`, never a recorder double. No network, no phone.

Fakes follow tests/test_action_label_keys.py / test_outcome_coverage.py's
shape (FakeJudge/FakeTransport).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from typesymbolic.judge import AskResult
from typesymbolic.question import Answer

from jevdevice.actions import ui
from jevdevice.budget import JEV_PROFILE, current_profile, profile_for
from jevdevice.calibrate import continuous as cont
from jevdevice.calibrate.units import label_case, store
from jevdevice.execution import dispatch
from jevdevice.journal import decision_log, outcomes

# --- fakes (same shape as test_action_label_keys.py / test_outcome_coverage.py) --


class FakeJudge:
    """Scripted ask_all(): pops one AskResult per call."""

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
    """Canned adb results: command -> RunResult, plus one XML dump."""

    name = "fake-device"

    def __init__(self, results: dict[str, RunResult], dump_xml: str = "") -> None:
        self._results = results
        self._dump_xml = dump_xml

    async def run(self, command: str, timeout: float = 15.0) -> RunResult:
        return self._results.get(command, RunResult())

    async def dump_hierarchy(self) -> str:
        return self._dump_xml

    async def window_size(self) -> tuple[int, int]:
        return 1080, 1920


def _choice(choice: str, probabilities: dict[str, float], confidence: float = 0.9) -> Answer:
    return Answer.from_choice(qid="", choice=choice, probabilities=probabilities, confidence=confidence)


def _noul(noul: float) -> Answer:
    return Answer.from_noul(qid="", noul=noul)


def _journal():
    return decision_log.get_journal()


def _pairs(group: str, scale: str) -> list[tuple[float, bool]]:
    return _journal().labeled_pairs(group, scale, engine="jev", any_revision=True)


@pytest.fixture(autouse=True)
def _clear_element_cache():
    # ui._ELEMENT_CACHE is a module-level LRU shared across every test file --
    # a full-suite run order can otherwise cache-hit a goal this file also uses.
    ui._ELEMENT_CACHE.clear()
    yield
    ui._ELEMENT_CACHE.clear()


# =============================================================================
# 1. Each kind through the REAL dispatch.run_kind / _run_ungated path
# =============================================================================

_TWO_BUTTON_DUMP = (
    '<hierarchy>'
    '<node text="OK" resource-id="" content-desc="" clickable="true" bounds="[0,0][100,50]"/>'
    '<node text="Cancel" resource-id="" content-desc="" clickable="true" bounds="[0,50][100,100]"/>'
    '</hierarchy>'
)
_ONE_BUTTON_DUMP = (
    '<hierarchy><node text="OK" resource-id="" content-desc="" clickable="true" bounds="[0,0][100,50]"/></hierarchy>'
)


async def test_fused_tap_through_run_kind_labels_pick_fit_gate() -> None:
    judge = FakeJudge("jev", [
        {  # fused pick + per-candidate safe noul
            "pick": _choice("text='Cancel'", {"text='OK'": 0.05, "text='Cancel'": 0.9}),
            "fit_0": _noul(0.2), "fit_1": _noul(0.9),
            "safe_0": _noul(0.5), "safe_1": _noul(0.9),
        },
        {"satisfied": _noul(0.9)},  # post-tap verify
    ])
    transport = FakeTransport({}, dump_xml=_TWO_BUTTON_DUMP)

    assert _pairs("pick", "confidence") == []
    assert _pairs("fit", "noul_p") == []
    assert _pairs("gate", "noul_p") == []

    response = await dispatch.run_kind(judge, transport, "tap", "press cancel")

    assert response["status"] == "ok"
    assert len(_pairs("pick", "confidence")) == 1
    assert len(_pairs("fit", "noul_p")) == 1
    assert len(_pairs("gate", "noul_p")) == 1


async def test_fallback_tap_through_run_kind_labels_pick_fit_gate(monkeypatch) -> None:
    # Force the fallback branch (options > chunk_size), same trick as
    # test_action_label_keys.py's fallback-tap test.
    tiny_chunk = replace(current_profile("jev"), chunk_size=0)
    monkeypatch.setattr(ui, "current_profile", lambda name=None: tiny_chunk)

    judge = FakeJudge("jev", [
        {"pick": _choice("text='OK'", {"text='OK'": 0.9}), "fit_0": _noul(0.9)},  # narrow_and_pick's own ask
        {"safe": _noul(0.9)},  # the separate gate_command ask
        {"satisfied": _noul(0.9)},  # post-tap verify
    ])
    transport = FakeTransport({}, dump_xml=_ONE_BUTTON_DUMP)

    response = await dispatch.run_kind(judge, transport, "tap", "press ok")

    assert response["status"] == "ok"
    assert len(_pairs("pick", "confidence")) == 1
    assert len(_pairs("fit", "noul_p")) == 1
    assert len(_pairs("gate", "noul_p")) == 1  # separate "safe" ask, still labeled


_TWO_FIELD_DUMP = (
    '<hierarchy>'
    '<node text="" resource-id="field_a" content-desc="" class="android.widget.EditText" '
    'clickable="true" bounds="[0,0][100,50]"/>'
    '<node text="" resource-id="field_b" content-desc="" class="android.widget.EditText" '
    'clickable="true" bounds="[0,50][100,100]"/>'
    '</hierarchy>'
)


async def test_type_text_through_run_kind_labels_pick_fit_gate() -> None:
    judge = FakeJudge("jev", [
        {  # fused field-narrow + value-pick
            "pick": _choice("resource-id='field_b'", {"resource-id='field_a'": 0.05, "resource-id='field_b'": 0.9}),
            "fit_0": _noul(0.2), "fit_1": _noul(0.9),
            "value": _choice("hello", {"hello": 0.9}), "any_fit_value": _noul(0.9),
        },
        {"safe": _noul(0.9)},  # gate_command's own separate ask
        {"satisfied": _noul(0.9)},  # post-type verify
    ])
    transport = FakeTransport({}, dump_xml=_TWO_FIELD_DUMP)

    response = await dispatch.run_kind(judge, transport, "type_text", "type 'hello' into field_b")

    assert response["status"] == "ok"
    assert len(_pairs("pick", "confidence")) == 1
    assert len(_pairs("fit", "noul_p")) == 1
    assert len(_pairs("gate", "noul_p")) == 1


async def test_keyevent_through_run_kind_labels_pick_and_gate_no_fit() -> None:
    # VOLUME_UP (not KEYCODE_HOME) is a real gate ask -- read-only classification skips only HOME.
    judge = FakeJudge("jev", [
        {"pick": _choice("VOLUME_UP", {"VOLUME_UP": 0.9}), "any_fit": _noul(0.9)},
        {"safe": _noul(0.9)},
    ])
    transport = FakeTransport({"input keyevent KEYCODE_VOLUME_UP": RunResult(exit_code=0)})

    response = await dispatch.run_kind(judge, transport, "keyevent", "turn the volume up")

    assert response["status"] == "ok"
    assert len(_pairs("pick", "confidence")) == 1
    assert len(_pairs("gate", "noul_p")) == 1
    assert _pairs("fit", "noul_p") == []  # closed-set kind: no fit noul ever asked


async def test_keyevent_nonzero_exit_code_labels_failed() -> None:
    """The one real failure producer verdict_from_response yields: a
    non-ok status whose response also carries a non-(None,0) exit_code."""
    judge = FakeJudge("jev", [
        {"pick": _choice("VOLUME_UP", {"VOLUME_UP": 0.9}), "any_fit": _noul(0.9)},
        {"safe": _noul(0.9)},
    ])
    transport = FakeTransport({"input keyevent KEYCODE_VOLUME_UP": RunResult(exit_code=1)})

    response = await dispatch.run_kind(judge, transport, "keyevent", "turn the volume up")

    assert response["status"] == "unverified"
    assert response["exit_code"] == 1
    pick_pairs = _pairs("pick", "confidence")
    gate_pairs = _pairs("gate", "noul_p")
    assert len(pick_pairs) == 1 and pick_pairs[0][1] is False
    assert len(gate_pairs) == 1 and gate_pairs[0][1] is False


async def test_toggle_service_through_run_kind_labels_pick_and_gate_no_fit() -> None:
    judge = FakeJudge("jev", [
        {
            "service": _choice("bluetooth", {"bluetooth": 0.95, "nfc": 0.03, "data": 0.02}),
            "enabled": _choice("on", {"on": 0.95, "off": 0.05}),
            "names_one": _noul(0.95),
        },
        {"safe": _noul(0.9)},  # gate_command's ask
        {  # execute_toggle's resolve-status narrow
            "pick": _choice("bluetooth_state", {"bluetooth_state": 0.9}),
            "fit_0": _noul(0.9),
        },
        {"satisfied": _noul(0.9)},  # execute_toggle's verify ask
    ])
    transport = FakeTransport({
        "svc bluetooth enable": RunResult(exit_code=0),
        "dumpsys -l": RunResult(stdout="Currently running services:\n  bluetooth_state\n"),
    })

    response = await dispatch.run_kind(judge, transport, "toggle_service", "turn on bluetooth")

    assert response["status"] == "ok"
    assert len(_pairs("pick", "confidence")) == 1  # the "service" key, not "enabled"
    assert len(_pairs("gate", "noul_p")) == 1
    assert _pairs("fit", "noul_p") == []  # ToggleProposal never asks a fit noul


async def test_open_app_through_run_kind_labels_pick_and_fit_no_gate() -> None:
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
        {"satisfied": _noul(0.9)},  # verify_with_retry's ask
    ])

    response = await dispatch.run_kind(judge, transport, "open_app", "open the calculator")

    assert response["status"] == "ok"
    assert len(_pairs("pick", "confidence")) == 1
    assert len(_pairs("fit", "noul_p")) == 1
    assert _pairs("gate", "noul_p") == []  # ungated kind: never asks safe


async def test_scroll_to_find_through_run_kind_labels_pick_and_fit_no_gate() -> None:
    dump_xml = (
        '<hierarchy><node text="Battery saver" resource-id="" content-desc="" '
        'clickable="true" bounds="[0,0][100,50]"/></hierarchy>'
    )
    transport = FakeTransport({}, dump_xml=dump_xml)
    judge = FakeJudge("jev", [{
        "pick": _choice("text='Battery saver'", {"text='Battery saver'": 0.9}),
        "fit_0": _noul(0.9),
    }])

    response = await dispatch.run_kind(judge, transport, "scroll_to_find", "find battery saver")

    assert response["status"] == "ok"
    assert len(_pairs("pick", "confidence")) == 1
    assert len(_pairs("fit", "noul_p")) == 1
    assert _pairs("gate", "noul_p") == []


# =============================================================================
# 2. Approval resume: on_pending hook labels the pick/gate call_ids, not a
#    key named after the kind (mirrors mcp_server.device_approve).
# =============================================================================


async def test_approval_resume_labels_pick_and_gate_not_the_kind_name() -> None:
    judge = FakeJudge("jev", [
        {  # toggle fill: confident pick, below-threshold gate below
            "service": _choice("bluetooth", {"bluetooth": 0.95, "nfc": 0.03, "data": 0.02}),
            "enabled": _choice("on", {"on": 0.95, "off": 0.05}),
            "names_one": _noul(0.95),
        },
        {"safe": _noul(0.5)},  # below the 0.8 gate threshold -> NEEDS_APPROVAL
        {  # (post-approval) execute_toggle's resolve-status narrow
            "pick": _choice("bluetooth_state", {"bluetooth_state": 0.9}),
            "fit_0": _noul(0.9),
        },
        {"satisfied": _noul(0.9)},  # execute_toggle's verify ask
    ])
    transport = FakeTransport({
        "svc bluetooth enable": RunResult(exit_code=0),
        "dumpsys -l": RunResult(stdout="Currently running services:\n  bluetooth_state\n"),
    })

    async def approve_hook(goal, kind, resume_arg, confidence, pending, verify, *, label):
        # Mirrors mcp_server.device_approve: run the already-gated command,
        # record the device verdict against run_kind's own LabelTarget.
        handler = dispatch.KIND_TABLE[kind]

        @dataclass
        class ResumeProposal:
            element: object
            service: object
            confidence: float

        resume_proposal = ResumeProposal(element=resume_arg, service=resume_arg, confidence=confidence)
        outcome = await handler.execute(judge, transport, goal, resume_proposal, pending.command, verify=verify, verbose=False)
        response = dispatch.response_for(kind, outcome, judge.name)
        outcomes.record_action(
            device=transport, label=label, gate=pending.gate_result,
            executed_command=pending.command.command, response=response,
        )
        return response

    response = await dispatch.run_kind(
        judge, transport, "toggle_service", "turn on bluetooth", on_pending=approve_hook,
    )

    assert response["status"] == "ok"
    assert len(_pairs("pick", "confidence")) == 1
    assert len(_pairs("gate", "noul_p")) == 1
    # No calibration unit a "toggle_service"-keyed verdict could accidentally hit.
    assert _pairs("fit", "noul_p") == []


# =============================================================================
# 3. Loop closure: seeded labels promote a tightening; current_profile()
# reads the promoted value; profile_for() stays pure defaults.
# =============================================================================
# Seeded via calibrate.units.label_case (real tagged rows) rather than 25+
# real run_kind calls per knob -- that would add no coverage beyond part 1.


def _seed_tightening(knob: str) -> None:
    journal = _journal()
    plan = [(0.95, True)] * 25 + [(0.85, False)] * 5
    for i, (value, correct) in enumerate(plan):
        label_case(journal, call_id=f"{knob}-{i}", engine="jev", knob=knob, value=value, correct=correct)


async def test_continuous_run_promotes_a_tightening_and_current_profile_reads_it() -> None:
    _seed_tightening("gate_threshold")
    _seed_tightening("min_fit")

    results = cont.run(engine="jev", human_signoff=False, quiet=True)

    gate_result = results["gate_threshold"]
    assert gate_result.applied is True
    assert gate_result.after > gate_result.before
    assert gate_result.before == JEV_PROFILE.gate_threshold

    fit_result = results["min_fit"]
    assert fit_result.applied is True
    assert fit_result.after > fit_result.before
    assert fit_result.before == JEV_PROFILE.min_fit

    live = current_profile("jev")
    assert live.gate_threshold == gate_result.after
    assert live.min_fit == fit_result.after
    # profile_for() is recalibration's own pure-default read, never overlaid.
    assert profile_for("jev").gate_threshold == JEV_PROFILE.gate_threshold
    assert profile_for("jev").min_fit == JEV_PROFILE.min_fit


async def test_continuous_run_never_loosens_without_signoff() -> None:
    journal = _journal()
    # All 30 labels correct at 0.55: the loosest noul_p candidate (0.50)
    # already clears precision, a genuine loosening below the 0.8 default.
    for i in range(30):
        label_case(journal, call_id=f"loosen-{i}", engine="jev", knob="gate_threshold", value=0.55, correct=True)

    results = cont.run(engine="jev", human_signoff=False, quiet=True)
    result = results["gate_threshold"]

    assert result.before == JEV_PROFILE.gate_threshold
    assert result.after >= result.before  # never loosened
    assert result.applied is False

    assert current_profile("jev").gate_threshold == JEV_PROFILE.gate_threshold
    assert store().get("gate|noul_p", engine="jev", model_revision=None, default=JEV_PROFILE.gate_threshold) == JEV_PROFILE.gate_threshold
