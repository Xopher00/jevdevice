"""Phase 9 offline tests: the Device protocol + AdbDevice adapter, the ACI
mapping on observations, and the METR-shaped phone task family. All zero
model calls, zero device -- the engine already has fake-transport coverage in
the untouched suite (test_outcome_coverage, test_recipes_planner), which is
itself the proof that the protocol is duck-compatible with the fakes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import jevdevice.device as device_mod
from jevdevice.device import AdbDevice, Device
from jevdevice.elements import Element, parse_editable_elements

PHASES_DIR = Path(__file__).resolve().parent.parent / "eval" / "phases"
if str(PHASES_DIR) not in sys.path:
    sys.path.insert(0, str(PHASES_DIR))


# --- T1: the protocol's surface is exactly the audited members -----------------

def test_device_protocol_carries_only_the_audited_members() -> None:
    """The roadmap sketch lists describe/risk_class/fits/settle too -- none has
    a current call site, so none may enter the protocol (zero hypothetical
    features). This pins that: adding one back requires a real consumer first."""
    members = {name for name in Device.__dict__ if not name.startswith("_")}
    assert members == {"dump_hierarchy", "run", "run_binary", "window_size"}
    assert "name" in Device.__annotations__


def test_engine_modules_import_the_device_protocol_not_the_transport() -> None:
    """T2's done-when, pinned mechanically: device-touching engine modules type
    against the protocol; only the composition root (common) and AdbDevice
    import transport.py. gate/planner carry no device import at all."""
    device_typed = ["dispatch.py", "ui.py", "elements.py", "services.py", "app_launch.py", "outcomes.py"]
    device_free = ["gate.py", "planner.py"]
    src = Path(device_mod.__file__).resolve().parent
    for name in device_typed:
        text = (src / name).read_text()
        assert "from .transport import" not in text, name
        assert "from .device import" in text, name
    for name in device_free:
        text = (src / name).read_text()
        assert "transport" not in text, name


# --- T2: AdbDevice is a thin wrapper over the frozen transport ------------------

class _StubTransport:
    """Duck-shaped transport: proves AdbDevice is a pure delegating wrapper."""

    def __init__(self, serial: str = "emulator-5554") -> None:
        self.serial = serial
        self.calls: list[tuple[str, tuple]] = []

    async def dump_hierarchy(self) -> str:
        self.calls.append(("dump_hierarchy", ()))
        return "<hierarchy/>"

    async def run(self, command: str, timeout: float = 15.0):
        self.calls.append(("run", (command, timeout)))
        return SimpleNamespace(stdout="ok\n", stderr="", exit_code=0)

    async def run_binary(self, command: str, timeout: float = 15.0) -> bytes:
        self.calls.append(("run_binary", (command, timeout)))
        return b"png"

    async def window_size(self) -> tuple[int, int]:
        self.calls.append(("window_size", ()))
        return (1080, 2400)


async def test_adbdevice_is_a_device_delegating_to_the_frozen_transport() -> None:
    stub = _StubTransport()
    device = AdbDevice(stub)
    assert isinstance(device, Device)  # runtime_checkable structural check
    # The journal identity: Device.name, and for the adb family it IS the serial
    # (the historical attribute name existing callers read stays available).
    assert device.name == device.serial == "emulator-5554"
    assert await device.dump_hierarchy() == "<hierarchy/>"
    result = await device.run("dumpsys battery")
    assert result.exit_code == 0
    assert await device.run_binary("screencap -p") == b"png"
    assert await device.window_size() == (1080, 2400)
    assert stub.calls == [
        ("dump_hierarchy", ()), ("run", ("dumpsys battery", 15.0)),
        ("run_binary", ("screencap -p", 15.0)), ("window_size", ()),
    ]


# --- T3: ACI-aligned keys on the existing observation shape ---------------------

EDITABLE_XML = (
    '<hierarchy>'
    '<node class="android.widget.EditText" text="Compose email" resource-id="com.example:id/editor"'
    ' clickable="true" bounds="[10,20][300,80]"/>'
    '</hierarchy>'
)


def test_element_carries_aci_aligned_role_from_the_real_class() -> None:
    elements = parse_editable_elements(EDITABLE_XML)
    element = next(iter(elements.values()))
    assert element.role == "text field"  # the noun already phrased inside description
    assert element.description.startswith("the text field")
    # The other ACI keys are the existing fields: text, bounds (bbox), and the
    # parse family (interactable). No fields invented.


def test_element_role_defaults_to_generic_element() -> None:
    assert Element(5, 5, "[0,0][10,10]").role == "element"


# --- T4: METR-shaped phone task family over the frozen goals + journal ----------

def _outcome_row(goal_id: str, verification: str) -> dict:
    return {"type": "outcome", "goal_id": goal_id, "verification": verification}


def test_family_exposes_dev_half_tasks_only() -> None:
    import metr_family
    from splitguard import heldout_goal_ids

    tasks = metr_family.PhoneTaskFamily().get_tasks()
    assert tasks, "the frozen phone goals must yield dev tasks"
    overlap = set(tasks) & heldout_goal_ids()
    assert not overlap, f"held-out goals leaked into the family: {overlap}"


def test_instructions_are_the_frozen_goal_text_and_unknown_tasks_fail_closed() -> None:
    import metr_family

    family = metr_family.PhoneTaskFamily()
    tasks = family.get_tasks()
    first = next(iter(tasks))
    assert family.add_instructions(first) == tasks[first]
    with pytest.raises(KeyError):
        family.add_instructions("nonexistent-task")


def test_verify_scores_from_journaled_verified_outcomes() -> None:
    import metr_family

    family = metr_family.PhoneTaskFamily()
    task = next(iter(family.get_tasks()))
    verified = [_outcome_row(task, "verified")]
    assert family.verify(task, verified) == 1.0
    assert family.verify(task, [_outcome_row(task, "escalated")]) == 0.0
    assert family.verify(task, []) == 0.0
    assert metr_family.aggregate_scores([task], verified)[task] == 1.0


async def test_a_metr_task_runs_through_the_runtime_end_to_end(journal_recorder) -> None:
    """The family's task resolves through the REAL planner (tier 2: scripted
    judge + duck device) -- zero engine changes, exactly the adapter's job."""
    import metr_family

    from jevdevice.budget import NONE_OF_THESE
    from jevdevice.jev import ChoiceAnswer, NoulAnswer

    family = metr_family.PhoneTaskFamily()
    task = next(iter(family.get_tasks()))

    DUMP_XML = (
        '<hierarchy>'
        '<node package="com.camera" text="" content-desc="Shutter" clickable="true"'
        ' bounds="[0,0][100,50]"/>'
        '</hierarchy>'
    )

    class DuckDevice:
        async def dump_hierarchy(self):
            return DUMP_XML

        async def run(self, command, timeout=15.0):
            return SimpleNamespace(stdout="", stderr="", exit_code=0)

        async def run_binary(self, command, timeout=15.0):
            return b""

        async def window_size(self):
            return (100, 100)

        name = "offline-duck"

    async def executor(jev, device, kind, step_goal, **kw):
        return SimpleNamespace(kind=kind, goal=step_goal, status="verified")

    def kind_pick(kind="tap"):
        return {
            "kind": ChoiceAnswer(choice=kind, probabilities={kind: 0.9, NONE_OF_THESE: 0.05}, confidence=0.9),
            "any_fit": NoulAnswer(noul=0.9),
        }

    class ScriptedJudge:
        engine_name = "jev"

        def __init__(self):
            self.payloads = [
                {"satisfied": NoulAnswer(noul=0.1)},  # not done yet
                kind_pick(),
                {"satisfied": NoulAnswer(noul=0.95)},  # done
            ]

        async def ask(self, state, questions, **kw):
            return self.payloads.pop(0)

    result = await family.run_task(task, ScriptedJudge(), DuckDevice(),
                                   executor=executor, store=_EmptyStore())
    assert result.tier == 2 and result.status == "resolved"


class _EmptyStore:
    def get(self, goal_id):
        return None

    def best_match(self, goal):
        return None


@pytest.fixture()
def journal_recorder(monkeypatch):
    """Planner rows must not touch the real journal (same pattern as
    test_recipes_planner)."""
    from jevdevice import decision_log

    class LiveJournal:
        def __init__(self) -> None:
            self.outcomes: list[dict] = []

        def record_outcome(self, **row) -> None:
            self.outcomes.append(row)

    live = LiveJournal()
    monkeypatch.setattr(decision_log, "get_journal", lambda: live)
    return live