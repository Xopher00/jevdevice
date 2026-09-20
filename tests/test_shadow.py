"""P5 shadow mode: the second engine observes jev's asks, journals linked rows,
and its answers never go anywhere. All unit-level: fakes for the HTTP path and
the router (same shapes as test_decision_log / test_laya_backend), so nothing
here touches the network or loads the checkpoint."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from jevdevice import shadow as shadow_mode
from jevdevice.calibrate.continuous import select_window
from jevdevice.jev import JevClient, Noul
from jevdevice.laya_backend import MODEL, LayaClient


class RecordingJournal:
    def __init__(self) -> None:
        self.decisions: list[dict] = []
        self.outcomes: list[dict] = []

    def record_decision(self, **row) -> None:
        self.decisions.append(row)

    def record_outcome(self, **row) -> None:
        self.outcomes.append(row)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeHTTP:
    """Records the order of events shared with the fake shadow (deterministic
    ordering proof for after-vs-concurrent scheduling)."""

    def __init__(self, payload: dict, events: list | None = None, delay: float = 0.0) -> None:
        self.payload = payload
        self.bodies: list[dict] = []
        self.events = events if events is not None else []
        self.delay = delay

    async def post(self, url, headers=None, json=None):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.bodies.append(json)
        self.events.append("primary-post")
        return _FakeResponse(self.payload)

    async def aclose(self) -> None:
        pass


class FakeRouter:
    def __init__(self, payload: dict | None = None, error: Exception | None = None) -> None:
        self.payload = payload or {
            "answers": {"q1": {"type": "noul", "noul": 0.42, "confidence": 0.42}},
            "usage": {"input_tokens": 7, "output_tokens": 0},
            "routing": {"model": MODEL, "repo": "convaiinnovations/laya", "reason": "explicit"},
        }
        self.error = error
        self.calls: list[dict] = []

    def predict(self, state, questions, model=None, **_kw) -> dict:
        self.calls.append({"state": state, "questions": questions, "model": model})
        if self.error is not None:
            raise self.error
        return self.payload


class _ObservingLaya(LayaClient):
    """Marks the moment a shadow ask actually starts (for scheduling-order
    assertions); otherwise the real LayaClient over a FakeRouter."""

    def __init__(self, *args, events: list | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._events = events if events is not None else []

    async def ask(self, *args, **kwargs):
        self._events.append("shadow-start")  # at entry, before any await
        return await super().ask(*args, **kwargs)


_PRIMARY_PAYLOAD = {
    "answers": {"q1": {"type": "noul", "noul": 0.9, "confidence": 0.9}},
    "usage": {"input_tokens": 10, "output_tokens": 2},
}


def _client(journal, events: list | None = None, delay: float = 0.0) -> tuple[JevClient, _FakeHTTP]:
    client = JevClient("test-key", journal=journal)
    fake = _FakeHTTP(_PRIMARY_PAYLOAD, events=events, delay=delay)
    client._client = fake
    return client, fake


def _drain(client: JevClient) -> None:
    """Let scheduled shadow tasks run to completion inside the test."""
    return asyncio.gather(*client.shadow_tasks)


# --- attach -------------------------------------------------------------------

async def test_attach_gives_a_jev_primary_a_laya_shadow(monkeypatch) -> None:
    monkeypatch.delenv(shadow_mode.SHADOW_ENV, raising=False)
    client, _ = _client(RecordingJournal())
    shadow_mode.attach(client)
    assert client.shadow is not None
    assert client.shadow.engine_name == "laya"
    first = client.shadow
    shadow_mode.attach(client)  # idempotent
    assert client.shadow is first


async def test_attach_skips_laya_primaries_and_the_off_switch(monkeypatch) -> None:
    monkeypatch.delenv(shadow_mode.SHADOW_ENV, raising=False)

    class _LayaPrimary:
        engine_name = "laya"
        shadow = None

    laya_primary = _LayaPrimary()
    shadow_mode.attach(laya_primary)
    assert laya_primary.shadow is None  # the shadow observes JEV traffic only

    monkeypatch.setenv(shadow_mode.SHADOW_ENV, "0")
    client, _ = _client(RecordingJournal())
    shadow_mode.attach(client)
    assert client.shadow is None


def test_unknown_shadow_mode_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv(shadow_mode.SHADOW_MODE_ENV, "side_by_side")
    with pytest.raises(SystemExit):
        shadow_mode.mode()


# --- scheduling + linkage -----------------------------------------------------

async def test_after_mode_shadows_once_the_answer_and_row_are_final() -> None:
    events: list[str] = []
    journal, shadow_journal = RecordingJournal(), RecordingJournal()
    client, _ = _client(journal, events=events)
    client.shadow = _ObservingLaya(journal=shadow_journal, router=FakeRouter(), events=events)

    answers = await client.ask({"goal": "press 7"}, {"q1": Noul(instructions="i")}, phase="verify")
    assert answers["q1"].noul == 0.9  # the primary answer, untouched
    assert events == ["primary-post"]  # nothing shadow-side ran during the ask
    await _drain(client)
    assert events == ["primary-post", "shadow-start"]

    primary, shadowed = journal.decisions[0], shadow_journal.decisions[0]
    assert primary["shadow_of"] is None and primary["engine"] == "jev"
    assert shadowed["shadow_of"] == primary["call_id"]  # the linkage
    assert shadowed["engine"] == "laya" and shadowed["phase"] == "verify"
    assert isinstance(primary["elapsed_ms"], float) and isinstance(shadowed["elapsed_ms"], float)


async def test_concurrent_mode_starts_the_shadow_before_the_answer_lands() -> None:
    events: list[str] = []

    journal, shadow_journal = RecordingJournal(), RecordingJournal()
    client, _fake = _client(journal, events=events, delay=0.02)  # primary yields first
    client.shadow = _ObservingLaya(journal=shadow_journal, router=FakeRouter(), events=events)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(shadow_mode.SHADOW_MODE_ENV, "concurrent")
        await client.ask({"goal": "g"}, {"q1": Noul(instructions="i")}, phase="recall")
    await _drain(client)
    assert events == ["shadow-start", "primary-post"]  # the shadow beat the answer
    assert shadow_journal.decisions[0]["shadow_of"] == journal.decisions[0]["call_id"]


async def test_shadow_failure_never_breaks_the_primary(monkeypatch) -> None:
    monkeypatch.delenv(shadow_mode.SHADOW_MODE_ENV, raising=False)
    journal, shadow_journal = RecordingJournal(), RecordingJournal()
    client, _ = _client(journal)
    client.shadow = LayaClient(journal=shadow_journal, router=FakeRouter(error=RuntimeError("laya blew up")))

    answers = await client.ask({"goal": "g"}, {"q1": Noul(instructions="i")}, phase="gate")
    await _drain(client)
    assert answers["q1"].noul == 0.9  # the primary path is whole
    shadowed = shadow_journal.decisions[0]
    assert shadowed["answers"] is None and "RuntimeError" in shadowed["error"]
    assert shadowed["shadow_of"] == journal.decisions[0]["call_id"]  # even the failure pairs up


async def test_shadow_answers_go_nowhere() -> None:
    journal, shadow_journal = RecordingJournal(), RecordingJournal()
    client, _ = _client(journal)
    client.shadow = LayaClient(journal=shadow_journal, router=FakeRouter())

    answers = await client.ask({"goal": "g"}, {"q1": Noul(instructions="i")}, phase="recall")
    await _drain(client)
    # The returned answers are the PRIMARY's, byte for byte; the shadow's
    # (0.42) exist only in its own journal row, for the report to read.
    assert answers["q1"].noul == 0.9
    assert shadow_journal.decisions[0]["answers"]["q1"]["noul"] == 0.42
    assert len(journal.decisions) == 1  # one primary row; no shadow pollution


async def test_aclose_drains_pending_shadow_tasks(monkeypatch) -> None:
    monkeypatch.delenv(shadow_mode.SHADOW_MODE_ENV, raising=False)
    journal, shadow_journal = RecordingJournal(), RecordingJournal()
    client, _ = _client(journal)
    client.shadow = LayaClient(journal=shadow_journal, router=FakeRouter())

    await client.ask({"goal": "g"}, {"q1": Noul(instructions="i")}, phase="recall")
    assert client.shadow_tasks  # scheduled, not yet run
    await client.aclose()
    shadowed = shadow_journal.decisions[0]
    assert shadowed["shadow_of"] == journal.decisions[0]["call_id"]  # drained, not dropped
    assert not [t for t in client.shadow_tasks if not t.done()]


# --- the P4.5 window must never see shadow rows -------------------------------

def test_p45_window_excludes_shadow_rows() -> None:
    now = datetime.now().astimezone()  # tz-aware: row ts are tz-aware isoformat
    primary = {"type": "decision", "ts": now.isoformat(), "call_id": "p1",
               "engine": "jev", "phase": "gate", "shadow_of": None}
    shadowed = {"type": "decision", "ts": now.isoformat(), "call_id": "s1",
                "engine": "laya", "phase": "gate", "shadow_of": "p1"}
    stale = {"type": "decision", "ts": (now - timedelta(days=90)).isoformat(),
             "call_id": "old", "engine": "laya", "phase": "gate", "shadow_of": None}
    rows, provenance = select_window([primary, shadowed, stale], now=now, window_days=30)
    assert [r["call_id"] for r in rows] == ["p1"]
    assert provenance["shadow_rows_excluded"] == 1
