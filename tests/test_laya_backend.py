"""LayaClient behind the engine flag -- contract mirror of JevClient.

Unit-level only: a scripted fake router stands in for laya's Router (the same
payload shapes the real one returns), so these tests never load the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jevdevice.common import bootstrap
from jevdevice.jev import Choice, JevClient, JevError, Noul, NoulAnswer
from jevdevice.laya_backend import MODEL, REVISION, LayaClient

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

async def test_ask_returns_typed_answers_and_records_usage() -> None:
    client = LayaClient(journal=RecordingJournal(), router=FakeRouter())
    answers = await client.ask({"goal": "press 7"}, _questions(), phase="recall")
    assert isinstance(answers["any"], NoulAnswer)
    assert answers["any"].noul == 0.9
    assert answers["pick"].choice == "beta"
    assert answers["pick"].probabilities["beta"] == 0.8
    assert client.usage.snapshot().input_tokens == 42
    assert client.usage.snapshot().output_tokens == 0


async def test_wire_questions_serialized_exactly_like_jev() -> None:
    """The frozen wire shape: same serialization expression as JevClient.ask,
    explicit model routing, state passed through untouched."""
    questions = _questions()
    fake = FakeRouter()
    client = LayaClient(journal=RecordingJournal(), router=fake)
    state = {"goal": "press 7", "candidates": ["alpha", "beta"]}
    await client.ask(state, questions, phase="recall")
    call = fake.calls[0]
    assert call["model"] == MODEL
    assert call["state"] == state
    assert call["questions"] == {
        name: q.model_dump(mode="json", exclude_none=True) for name, q in questions.items()
    }


async def test_decision_row_carries_laya_engine_and_pinned_revision() -> None:
    recorder = RecordingJournal()
    client = LayaClient(journal=recorder, router=FakeRouter())
    await client.ask({"goal": "g"}, _questions(), phase="recall", call_id="cid-laya")
    row = recorder.decisions[0]
    assert row["engine"] == "laya"
    assert row["model_revision"] == REVISION
    assert row["phase"] == "recall"
    assert row["call_id"] == "cid-laya"
    assert row["answers"]["any"] == {"type": "noul", "noul": 0.9}  # full distribution as parsed (extra keys dropped, same as the Jev path)
    assert row["answers"]["pick"]["choice"] == "beta"
    assert row["error"] is None


async def test_budget_overflow_valueerror_maps_to_jev_error_and_journals() -> None:
    """Laya raises ValueError on budget overflow; the message may blame
    head_max_len even when max_len triggered it -- catch the type, never the text."""
    recorder = RecordingJournal()
    boom = ValueError("question 'pick' options exceed head_max_len=256")
    client = LayaClient(journal=recorder, router=FakeRouter(error=boom))
    with pytest.raises(JevError) as excinfo:
        await client.ask({"goal": "g"}, _questions(), phase="ground")
    assert "budget overflow" in str(excinfo.value)
    assert excinfo.value.__cause__ is boom
    row = recorder.decisions[0]
    assert row["answers"] is None
    assert "budget overflow" in row["error"]
    assert row["phase"] == "ground"


async def test_ask_requires_questions_like_jev() -> None:
    client = LayaClient(journal=RecordingJournal(), router=FakeRouter())
    with pytest.raises(JevError):
        await client.ask({"goal": "g"}, {})


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
    jev = JevClient("test-key")
    laya = LayaClient(router=FakeRouter())
    jev_snap = jev.usage.snapshot()
    laya_snap = laya.usage.snapshot()
    assert jev_snap.engine_info.engine == "jev"
    assert jev_snap.engine_info.model_revision == "jev-1.13.0"
    assert laya_snap.engine_info.engine == "laya"
    assert laya_snap.engine_info.model_revision == REVISION
    assert laya_snap.engine_info.model == MODEL
    assert laya_snap.engine_info.routing is None  # filled by the first ask


async def test_ledger_routing_metadata_updates_per_ask() -> None:
    fake = FakeRouter()
    client = LayaClient(journal=RecordingJournal(), router=fake)
    await client.ask({"goal": "g"}, _questions(), phase="recall")
    info = client.usage.snapshot().engine_info
    assert info is not None
    assert info.routing == fake.payload["routing"]
