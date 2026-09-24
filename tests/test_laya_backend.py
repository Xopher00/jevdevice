"""LayaClient behind the engine flag -- a plain typesymbolic JudgeEngine,
contract mirror of JevEngine's ask_all/usage shape.

Unit-level only: a scripted fake router stands in for laya's Router (the same
payload shapes the real one returns), so these tests never load the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typesymbolic.judge import JudgeError

from jevdevice.common import DEFAULT_MODEL, bootstrap
from jevdevice.jev import Choice, Noul, ask
from jevdevice.journal import decision_log
from jevdevice.laya_backend import MODEL, REVISION, LayaClient
from jevdevice.ledger import EngineInfo, UsageLedger

LAYA_BACKEND_SRC = (Path(__file__).resolve().parent.parent / "src" / "jevdevice" / "laya_backend.py").read_text(encoding="utf-8")


class RecordingJournal:
    def __init__(self) -> None:
        self.decisions: list[dict] = []
        self.outcomes: list[dict] = []

    def record_decision(self, **row) -> None:
        self.decisions.append(row)

    def record_outcome(self, **row) -> None:
        self.outcomes.append(row)


class FakeRouter:
    """Scripted Router.predict: records the wire body, returns the payload shape
    the real one returns (answers + usage + routing)."""

    def __init__(self, payload: dict | None = None, error: Exception | None = None) -> None:
        self.payload = payload or {
            "answers": {
                "any": {"type": "noul", "noul": 0.9, "confidence": 0.9},
                "pick": {
                    "type": "choice", "choice": "beta",
                    "probabilities": {"alpha": 0.1, "beta": 0.8, "gamma": 0.1},
                    "confidence": 0.8,
                },
            },
            "usage": {"input_tokens": 42, "output_tokens": 0},
            "routing": {"model": MODEL, "repo": "convaiinnovations/laya", "reason": "explicit"},
        }
        self.error = error
        self.calls: list[dict] = []

    def predict(self, state, questions, model=None, **_kw) -> dict:
        self.calls.append({"state": state, "questions": questions, "model": model})
        if self.error is not None:
            raise self.error
        return self.payload


def _questions() -> dict:
    return {
        "any": Noul(instructions="Could any of these candidates satisfy the goal?"),
        "pick": Choice(instructions="pick one", criteria={"alpha": "first", "beta": "second"}),
    }


# --- the ask() contract -------------------------------------------------------

async def test_ask_returns_typed_answers_and_records_usage(monkeypatch) -> None:
    monkeypatch.setattr(decision_log, "_default_journal", RecordingJournal())
    client = LayaClient(router=FakeRouter())
    _, answers = await ask(client, {"goal": "press 7"}, _questions(), phase="recall")
    assert answers["any"].type == "noul"
    assert answers["any"].noul == 0.9
    assert answers["pick"].choice == "beta"
    assert answers["pick"].probabilities["beta"] == 0.8
    assert client.usage.snapshot().input_tokens == 42
    assert client.usage.snapshot().output_tokens == 0


async def test_wire_questions_serialized_exactly_like_jev(monkeypatch) -> None:
    """The frozen wire shape: same serialization expression as JevEngine,
    explicit model routing, state passed through untouched."""
    monkeypatch.setattr(decision_log, "_default_journal", RecordingJournal())
    questions = _questions()
    fake = FakeRouter()
    client = LayaClient(router=fake)
    state = {"goal": "press 7", "candidates": ["alpha", "beta"]}
    await ask(client, state, questions, phase="recall")
    call = fake.calls[0]
    assert call["model"] == MODEL
    assert call["state"] == state
    assert call["questions"] == {
        name: q.model_dump(mode="json", exclude_none=True) for name, q in questions.items()
    }


async def test_decision_row_carries_laya_engine_and_pinned_revision(monkeypatch) -> None:
    recorder = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", recorder)
    client = LayaClient(router=FakeRouter())
    call_id, _ = await ask(client, {"goal": "g"}, _questions(), phase="recall")
    row = recorder.decisions[0]
    assert row["engine"] == "laya"
    assert row["model_revision"] == REVISION
    assert row["phase"] == "recall"
    assert row["call_id"] == call_id
    assert row["answers"]["any"].noul == 0.9  # full distribution as parsed, same as the Jev path
    assert row["answers"]["pick"].choice == "beta"
    assert row.get("error") is None


async def test_budget_overflow_valueerror_maps_to_judge_error_and_journals(monkeypatch) -> None:
    """Laya raises ValueError on budget overflow; the message may blame
    head_max_len even when max_len triggered it -- catch the type, never the text."""
    recorder = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", recorder)
    boom = ValueError("question 'pick' options exceed head_max_len=256")
    client = LayaClient(router=FakeRouter(error=boom))
    with pytest.raises(JudgeError) as excinfo:
        await ask(client, {"goal": "g"}, _questions(), phase="ground")
    assert "budget overflow" in str(excinfo.value)
    assert excinfo.value.__cause__ is boom
    row = recorder.decisions[0]
    assert row["answers"] == {}
    assert "budget overflow" in row["error"]
    assert row["phase"] == "ground"


async def test_ask_requires_questions_like_jev(monkeypatch) -> None:
    monkeypatch.setattr(decision_log, "_default_journal", RecordingJournal())
    client = LayaClient(router=FakeRouter())
    with pytest.raises(JudgeError):
        await ask(client, {"goal": "g"}, {})


def test_checkpoint_resolves_from_the_local_cache_first() -> None:
    """The pinned snapshot must load offline-first: once it's in the HF cache,
    no process start pays a network round trip (the online fetch runs only if
    the cache is cold)."""
    assert "local_files_only=True" in LAYA_BACKEND_SRC


def test_no_httpx_in_the_backend_module() -> None:
    """The backend module must not grow a network dependency (an 'import httpx'
    anywhere; the word in prose/comments is fine)."""
    assert "import httpx" not in LAYA_BACKEND_SRC


# --- the engine flag ----------------------------------------------------------

def test_bootstrap_laya_engine_needs_no_typesafe_key(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_AI_API", raising=False)
    monkeypatch.setenv("JEV_ENGINE", "laya")
    client, transport = bootstrap(serial="test-serial")
    assert isinstance(client, LayaClient)
    assert transport.serial == "test-serial"


def test_bootstrap_jev_engine_requires_the_key(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_AI_API", raising=False)
    monkeypatch.setattr("jevdevice.common.load_env_file", lambda *a, **k: None)  # pin: no .env on this machine
    monkeypatch.setenv("JEV_ENGINE", "jev")
    with pytest.raises(SystemExit):
        bootstrap(serial="test-serial")


def test_bootstrap_default_engine_is_jev(monkeypatch) -> None:
    monkeypatch.delenv("JEV_ENGINE", raising=False)
    monkeypatch.delenv("TYPESAFE_AI_API", raising=False)
    monkeypatch.setattr("jevdevice.common.load_env_file", lambda *a, **k: None)  # pin: no .env on this machine
    with pytest.raises(SystemExit):  # default engine demands the key
        bootstrap(serial="test-serial")


def test_bootstrap_rejects_unknown_engine(monkeypatch) -> None:
    monkeypatch.setenv("JEV_ENGINE", "gpt")
    with pytest.raises(SystemExit):
        bootstrap(serial="test-serial")


# --- ledger snapshots distinguish the engines ----------------------------------

def test_ledger_distinguishes_engines() -> None:
    from typesymbolic.judge import JevEngine

    jev = JevEngine(api_key="test-key", model=DEFAULT_MODEL)
    jev.usage = UsageLedger()
    jev.usage.record_engine(EngineInfo(engine="jev", model_revision=DEFAULT_MODEL))
    laya = LayaClient(router=FakeRouter())
    jev_snap = jev.usage.snapshot()
    laya_snap = laya.usage.snapshot()
    assert jev_snap.engine_info.engine == "jev"
    assert jev_snap.engine_info.model_revision == DEFAULT_MODEL
    assert laya_snap.engine_info.engine == "laya"
    assert laya_snap.engine_info.model_revision == REVISION
    assert laya_snap.engine_info.model == MODEL
    assert laya_snap.engine_info.routing is None  # filled by the first ask


async def test_ledger_routing_metadata_updates_per_ask(monkeypatch) -> None:
    monkeypatch.setattr(decision_log, "_default_journal", RecordingJournal())
    fake = FakeRouter()
    client = LayaClient(router=fake)
    await ask(client, {"goal": "g"}, _questions(), phase="recall")
    info = client.usage.snapshot().engine_info
    assert info is not None
    assert info.routing == fake.payload["routing"]
