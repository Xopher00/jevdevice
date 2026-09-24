"""CliDevice (the local-shell task family) through the UNTOUCHED engine.
Zero model calls; the only subprocesses are fixed read-only shell commands
(echo/printf/pwd/ls), mirroring what the adapter itself owns.

The proof structure mirrors test_device_protocol.py: a runtime_checkable
structural check, delegation with recorded calls, and an engine-level test --
here, that the phone-closed engine degrades fail-closed on a CLI device
(snapshot isn't a UI tree -> ungrounded kind pick; every CLI goal abstains
over ACTION_KINDS -> escalate).
"""

from __future__ import annotations

import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
from typesymbolic.journal import Journal
from typesymbolic.judge import ScriptedJudge
from typesymbolic.question import Answer

from jevdevice import question_sets
from jevdevice.device import CLI_SNAPSHOT_COMMAND, CliDevice, Device
from jevdevice.execution.dispatch import pick_kind


@pytest.fixture()
def cli_device() -> CliDevice:
    return CliDevice()


# --- the adapter is a Device, with its own journal identity ---------------------

async def test_clidevice_is_a_device_with_a_stable_family_name(cli_device: CliDevice) -> None:
    assert isinstance(cli_device, Device)  # runtime_checkable structural check
    assert cli_device.name == f"cli:{platform.node()}"
    # Distinct from every adb identity: the journal's device column separates
    # the families by construction, not by query discipline.
    assert cli_device.name.startswith("cli:")


# --- the four members, over the real frozen LocalShellTransport -----------------

async def test_run_returns_exit_code_and_stdout(cli_device: CliDevice) -> None:
    result = await cli_device.run("printf cli-ok")
    assert result.exit_code == 0
    assert result.stdout == "cli-ok"

    failing = await cli_device.run("false")
    assert failing.exit_code != 0  # the exit code is the device's own truth


async def test_run_binary_captures_raw_stdout_bytes(cli_device: CliDevice) -> None:
    assert await cli_device.run_binary("printf raw") == b"raw"
    with pytest.raises(RuntimeError):
        await cli_device.run_binary("sh -c 'echo err >&2; exit 3'")


async def test_window_size_is_terminal_geometry(cli_device: CliDevice) -> None:
    width, height = await cli_device.window_size()
    assert width > 0 and height > 0  # never consumed on CLI paths; honest, not fake phone pixels


async def test_dump_hierarchy_is_a_plain_text_snapshot(cli_device: CliDevice) -> None:
    snapshot = await cli_device.dump_hierarchy()
    assert CLI_SNAPSHOT_COMMAND in snapshot  # the fixed read-only command the adapter owns
    # NOT a UI tree: the engine's XML parser rejects it, and that rejection is
    # the load-bearing fail-closed behavior the kind pick relies on.
    with pytest.raises(ET.ParseError):
        ET.fromstring(snapshot)


# --- the untouched engine degrades fail-closed on the second family -------------

def _abstaining_judge() -> ScriptedJudge:
    """The kind pick abstains over the phone-closed ACTION_KINDS -- what EVERY
    CLI goal must do (fail-closed escalation)."""
    return ScriptedJudge([{
        "kind": Answer.from_choice(qid="", choice="none_of_these",
                             probabilities={"none_of_these": 0.9}, confidence=0.9),
        "any_fit": Answer.from_noul(qid="", noul=0.1),
    }])


async def test_pick_kind_degrades_to_ungrounded_then_escalates(cli_device: CliDevice) -> None:
    judge = _abstaining_judge()
    pick = await pick_kind(judge, "What is the CPU temperature right now?", cli_device, verbose=False)
    # The engine asked with the UNGROUNDED compiled wording: the CLI snapshot
    # is not a UI tree, so screen grounding degraded (fail-closed) instead of
    # lying about a screen.
    assert len(judge.calls) == 1
    state, questions = judge.calls[0]
    assert questions["kind"].instructions == question_sets.text("kind.pick")
    assert "screen" not in state
    # And the phone-closed vocabulary abstains: escalate, never guess.
    assert pick.kind is None
    assert any("abstained" in reason for reason in pick.reasons)

# --- the gated-probe runner, offline: scripted judge, recorded journal ----------

PHASES_DIR = Path(__file__).resolve().parent.parent / "eval" / "phases"
if str(PHASES_DIR) not in sys.path:
    sys.path.insert(0, str(PHASES_DIR))

import cli_family


class _StubCliTransport:
    """Records run() calls without executing -- proves pending never executes."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    async def run(self, command: str, timeout: float = 15.0):
        self.ran.append(command)
        return SimpleNamespace(stdout="6.8.0-generic\n", stderr="", exit_code=0)


@pytest.fixture()
def journal_recorder(monkeypatch, tmp_path: Path) -> Journal:
    from jevdevice.journal import decision_log

    journal = Journal(root=tmp_path, rotation="daily", background_writes=False)
    monkeypatch.setattr(decision_log, "get_journal", lambda: journal)
    return journal


@pytest.fixture()
def question_set_v2(monkeypatch):
    monkeypatch.setenv("JEV_QUESTION_SET", "v2")  # the family's frozen wordings
    from jevdevice import question_sets

    question_sets._cache.pop("v2", None)  # re-read under the test's env
    yield "v2"
    question_sets._cache.pop("v2", None)


def _pick_payload(choice: str) -> dict:
    return {
        "pick": Answer.from_choice(qid="", choice=choice,
                             probabilities={choice: 0.9}, confidence=0.9),
        "any_fit": Answer.from_noul(qid="", noul=0.9),
    }


async def test_run_probe_end_to_end_through_the_untouched_engine(
    cli_device: CliDevice, journal_recorder: Journal, question_set_v2,
) -> None:
    judge = ScriptedJudge([
        _pick_payload("uname -r"),                          # the closed-set pick
        {"safe": Answer.from_noul(qid="", noul=0.9)},        # the gate ask
        {"satisfied": Answer.from_noul(qid="", noul=0.9)},   # the verify ask
    ])
    response = await cli_family.run_probe(
        judge, cli_device, "What kernel version is running?", ["uname -r"], verbose=False)
    assert response["status"] == "ok" and response["exit_code"] == 0
    # three real asks: the closed-set pick, the safety gate, the verification
    assert [next(iter(questions)) for _state, questions in judge.calls] == ["pick", "safe", "satisfied"]
    assert judge.calls[1][1]["safe"].instructions == question_sets.text("cli.safe")
    # one outcome row, joined to its verdict by call_id, device column = the CLI family
    rows = list(journal_recorder.replay())
    outcome_rows = [r for r in rows if r["type"] == "outcome"]
    verdict_rows = [r for r in rows if r["type"] == "verdict"]
    assert len(outcome_rows) == 1
    row = outcome_rows[0]
    assert row["device"].startswith("cli:")
    assert row["kind"] == "cli_probe"
    assert len(verdict_rows) == 1
    assert verdict_rows[0]["status"] == "verified"
    assert verdict_rows[0]["call_id"] == row["call_id"]


async def test_run_probe_never_auto_approves(
    journal_recorder: Journal, question_set_v2,
) -> None:
    judge = ScriptedJudge([
        _pick_payload("uname -r"),
        {"safe": Answer.from_noul(qid="", noul=0.3)},  # below the gate threshold
    ])
    stub = _StubCliTransport()
    device = CliDevice(stub)
    response = await cli_family.run_probe(
        judge, device, "What kernel version is running?", ["uname -r"], verbose=False)
    assert response["status"] == "escalated"
    assert stub.ran == []  # the pending verdict NEVER executes
    rows = list(journal_recorder.replay())
    verdict_rows = [r for r in rows if r["type"] == "verdict"]
    assert len(verdict_rows) == 1 and verdict_rows[0]["status"] == "escalated"
