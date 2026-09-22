"""Recipe store + tiered planner, all offline: fakes for the judge, the
transport, and the per-step executor -- no device, no network. The
no-free-form guard is enforced structurally: the planner may only ask the
judge compiled questions or a closed-vocabulary choice.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from jevdevice.execution import planner
from jevdevice.execution import recipes as recipes_mod
from jevdevice.execution.recipes import (
    Recipe,
    RecipeStep,
    RecipeStore,
    embed,
    recipes_from_journal,
    similarity,
    verify_recipe,
)
from jevdevice.jev import ChoiceAnswer, NoulAnswer
from jevdevice.journal.decision_log import goal_id_for
from jevdevice.judge.gate import CommandVariant, GateResult


class RecordingJournal:
    """Duck-typed journal: replay() yields prebuilt rows."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def replay(self):
        yield from self._rows


class LiveJournal:
    """Records outcome rows without touching disk."""

    def __init__(self) -> None:
        self.outcomes: list[dict] = []

    def record_outcome(self, **row) -> None:
        self.outcomes.append(row)


class FakeJudge:
    """Scripted ask(): pops one answers payload per call."""

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []
        self.engine_name = "jev"

    async def ask(self, state, questions, **kw):
        self.calls.append({"state": state, "questions": questions, **kw})
        return self.payloads.pop(0)


def outcome_row(goal: str, *, kind: str | None, verification: str, command: str | None = None,
                edge: dict | None = None, call_id: str | None = None, ts: str = "") -> dict:
    return {
        "type": "outcome", "ts": ts, "call_id": call_id, "goal_id": goal_id_for(goal),
        "goal": goal, "device": "emulator", "executed_command": command,
        "verification": verification, "status": None, "recovery_command": None,
        "graph_edge": edge, "decision": None, "reasons": None, "exit_code": None,
        "satisfied": None, "executed": None, "kind": kind, "tier": None, "recipe_id": None,
    }


def _recipe(goal: str, steps: list[RecipeStep]) -> Recipe:
    return Recipe(recipe_id=goal_id_for(goal), goal=goal, goal_id=goal_id_for(goal),
                  steps=steps, embedding=embed(goal),
                  created="2026-09-21T00:00:00", question_set="v1")


@pytest.fixture()
def journal_recorder(monkeypatch):
    from jevdevice.journal import decision_log

    live = LiveJournal()
    monkeypatch.setattr(decision_log, "get_journal", lambda: live)
    return live


# --- goal-text embedding ---------------------------------------------------------

def test_embed_is_deterministic_and_normalized() -> None:
    assert embed("open the clock app") == embed("open the clock app")
    vec = embed("open the clock app")
    assert abs(sum(c * c for c in vec) - 1.0) < 1e-9


def test_similar_goals_rank_above_unrelated() -> None:
    query = embed("open the camera app and take a picture")
    near = similarity(query, embed("open the camera app"))
    far = similarity(query, embed("toggle bluetooth off"))
    assert near > far
    assert far < recipes_mod.RETRIEVAL_FLOOR  # unrelated goals never match


# --- builder: verified chains, dead ends and backtracking removed ---------------

def test_builder_keeps_only_the_final_verified_chain() -> None:
    goal = "set an alarm"
    rows = [
        outcome_row(goal, kind="open_app", verification="escalated", ts="2026-09-21T01:00:00"),
        outcome_row(goal, kind="tap", verification="failed", command="input tap 1 1", ts="2026-09-21T01:00:01"),
        outcome_row(goal, kind="open_app", verification="verified", command="monkey -p com.clock 1",
                    edge={"from_node": "home", "to_node": "com.clock"}, call_id="c1", ts="2026-09-21T01:01:00"),
        outcome_row(goal, kind="tap", verification="verified", command="input tap 5 700",
                    edge={"from_node": "com.clock", "to_node": "com.clock"}, call_id="c2", ts="2026-09-21T01:01:05"),
    ]
    recipe = recipes_from_journal(RecordingJournal(rows))[goal_id_for(goal)]
    assert [(s.kind, s.call_id) for s in recipe.steps] == [("open_app", "c1"), ("tap", "c2")]
    assert recipe.steps[0].to_node == "com.clock"


def test_builder_compresses_backtracked_duplicates() -> None:
    goal = "toggle wifi off then on"
    rows = [
        outcome_row(goal, kind="open_app", verification="verified", command="monkey -p com.settings 1", call_id="c1", ts="2026-09-21T02:00:00"),
        outcome_row(goal, kind="open_app", verification="verified", command="monkey -p com.settings 1", call_id="c2", ts="2026-09-21T02:00:10"),  # backtracked duplicate
        outcome_row(goal, kind="tap", verification="failed", command="input tap 1 1", call_id="c3", ts="2026-09-21T02:00:20"),  # dead end
        outcome_row(goal, kind="toggle_service", verification="verified", command="svc wifi disable", call_id="c4", ts="2026-09-21T02:00:30"),
    ]
    recipe = recipes_from_journal(RecordingJournal(rows))[goal_id_for(goal)]
    # the earlier duplicate open_app compresses to the last occurrence; the
    # failed tap never enters; the chain keeps last-occurrence order
    assert [(s.kind, s.command) for s in recipe.steps] == [
        ("open_app", "monkey -p com.settings 1"),
        ("toggle_service", "svc wifi disable"),
    ]


def test_builder_skips_goals_without_verified_kind_rows() -> None:
    rows = [
        outcome_row("never done", kind="tap", verification="escalated", ts="t"),
        outcome_row("old rows", kind=None, verification="verified", command="x", ts="t"),
    ]
    assert recipes_from_journal(RecordingJournal(rows)) == {}


def test_builder_refuses_heldout_goals() -> None:
    goal = "heldout only"
    rows = [outcome_row(goal, kind="tap", verification="verified", command="input tap 1 1", call_id="c", ts="t")]
    with pytest.raises(ValueError, match="held-out"):
        recipes_from_journal(RecordingJournal(rows), heldout_goal_ids={goal_id_for(goal)})


# --- store + replay ------------------------------------------------------------

def test_store_roundtrip_and_retrieval(tmp_path) -> None:
    recipe = _recipe("open the camera app", [
        RecipeStep("open_app", "open the camera app", "monkey -p com.camera 1", "home", "com.camera", "c1"),
    ])
    RecipeStore(tmp_path).upsert(recipe)
    reloaded = RecipeStore(tmp_path)
    assert reloaded.get(goal_id_for("open the camera app")) is not None
    hit = reloaded.best_match("open the camera app and take a picture")
    assert hit is not None and hit[0].recipe_id == recipe.recipe_id
    assert reloaded.best_match("toggle bluetooth off") is None  # below the floor


def test_stored_recipe_replays_to_its_verified_outcome() -> None:
    goal = "open the camera app"
    rows = [outcome_row(goal, kind="open_app", verification="verified", command="monkey -p com.camera 1", call_id="c1", ts="t")]
    good = _recipe(goal, [RecipeStep("open_app", goal, "monkey -p com.camera 1", "home", "com.camera", "c1")])
    drifted = _recipe(goal, [RecipeStep("open_app", goal, "monkey -p com.camera 1", "home", "com.camera", "cMISSING")])
    assert verify_recipe(good, RecordingJournal(rows))
    assert not verify_recipe(drifted, RecordingJournal(rows))


# --- tier 0: recipe hit ------------------------------------------------------------

async def test_tier0_recipe_hit_resolves(journal_recorder, tmp_path) -> None:
    goal = "open the camera app"
    recipe = _recipe(goal, [
        RecipeStep("open_app", goal, "monkey -p com.camera 1", "home", "com.camera", "c1"),
        RecipeStep("tap", goal, "input tap 5 700", "com.camera", "com.camera", "c2"),
    ])
    store = RecipeStore(tmp_path)
    store.upsert(recipe)
    ran: list[tuple[str, str]] = []

    async def executor(jev, transport, kind, step_goal, **kw):
        ran.append((kind, step_goal))
        return planner.StepResult(kind, step_goal, "verified")

    result = await planner.resolve(FakeJudge([]), None, goal, executor=executor, store=store)
    assert result.tier == 0 and result.status == "resolved"
    assert ran == [("open_app", goal), ("tap", goal)]
    assert [r["tier"] for r in journal_recorder.outcomes if r["status"] == "planner_resolved"] == [0]


# --- tier 1: bounded adapt (drop-only) ----------------------------------------------

async def test_tier1_adapts_by_dropping_the_stale_step(journal_recorder, tmp_path) -> None:
    goal = "open the camera app and take a picture"
    recipe = _recipe(goal, [
        RecipeStep("open_app", goal, "monkey -p com.camera 1", "home", "com.camera", "c1"),
        RecipeStep("tap", "tap the shutter", "input tap 5 700", "com.camera", "com.camera", "c2"),
        RecipeStep("toggle_service", "toggle flash", "svc flash off", "com.camera", "com.camera", "c3"),
    ])
    store = RecipeStore(tmp_path)
    store.upsert(recipe)
    ran: list[str] = []

    async def executor(jev, transport, kind, step_goal, **kw):
        ran.append(kind)
        # tier 0: the tap fails on the current screen; tier 1: everything it runs verifies
        if kind == "tap" and kw.get("tier") == 0:
            return planner.StepResult(kind, step_goal, "escalated")
        return planner.StepResult(kind, step_goal, "verified")

    judge = FakeJudge([
        {"fits": NoulAnswer(noul=0.9)},   # tap still fits -> re-run it
        {"fits": NoulAnswer(noul=0.1)},   # toggle no longer applies -> skip it
    ])
    result = await planner.resolve(judge, DumpTransport(), goal, executor=executor, store=store)
    assert result.tier == 1 and result.status == "resolved"
    assert ran == ["open_app", "tap", "tap"]  # tier 1 resumes from the failed step
    assert [s.kind for s in result.steps if s.status == "skipped"] == ["toggle_service"]
    statuses = [r["status"] for r in journal_recorder.outcomes]
    assert "planner_fallthrough" in statuses and "planner_resolved" in statuses


# --- tier 2: stepwise selection ------------------------------------------------------

DUMP_XML = (
    '<hierarchy>'
    '<node package="com.camera" text="" content-desc="Shutter" clickable="true" bounds="[0,0][100,50]"/>'
    '</hierarchy>'
)


class DumpTransport:
    async def dump_hierarchy(self) -> str:
        return DUMP_XML

    async def run(self, command: str, timeout: float = 15.0):
        return SimpleNamespace(stdout="", stderr="", exit_code=0)


def _kind_pick_payload(confidence: float = 0.9, kind: str | None = "tap"):
    from jevdevice.budget import NONE_OF_THESE

    if kind is None:  # the judge abstains
        return {
            "kind": ChoiceAnswer(choice=NONE_OF_THESE, probabilities={NONE_OF_THESE: confidence}, confidence=confidence),
            "any_fit": NoulAnswer(noul=0.9),
        }
    return {
        "kind": ChoiceAnswer(choice=kind, probabilities={kind: confidence, NONE_OF_THESE: 0.05}, confidence=confidence),
        "any_fit": NoulAnswer(noul=0.9),
    }


async def test_tier2_stepwise_selection_resolves(journal_recorder, tmp_path) -> None:
    ran: list[str] = []

    async def executor(jev, transport, kind, step_goal, **kw):
        ran.append(kind)
        return planner.StepResult(kind, step_goal, "verified")

    # not done -> pick tap (closed vocabulary) -> done.
    judge = FakeJudge([
        {"satisfied": NoulAnswer(noul=0.1)},
        _kind_pick_payload(),
        {"satisfied": NoulAnswer(noul=0.95)},
    ])
    result = await planner.resolve(judge, DumpTransport(), "tap the shutter button",
                                   executor=executor, store=RecipeStore(tmp_path))
    assert result.tier == 2 and result.status == "resolved"
    assert ran == ["tap"]
    assert [r["tier"] for r in journal_recorder.outcomes if r["status"] == "planner_resolved"] == [2]


async def test_tier2_verified_read_resolves_without_a_screen_change(journal_recorder, tmp_path) -> None:
    """Regression (planner batch evidence): a device-verified dumpsys step is
    the goal's deliverable -- reads leave the screen unchanged, so the loop
    must resolve on the verified read instead of re-running it to the step
    bound. The batch run ground `dumpsys battery` twelve verified times and
    still fell through cold."""
    ran: list[str] = []

    async def executor(jev, transport, kind, step_goal, **kw):
        ran.append(kind)
        return planner.StepResult(kind, step_goal, "verified")

    # not done on screen -> pick dumpsys -> the read verifies: NO further
    # satisfied ask may run (FakeJudge pops on empty would IndexError).
    judge = FakeJudge([
        {"satisfied": NoulAnswer(noul=0.1)},
        _kind_pick_payload(kind="dumpsys"),
    ])
    result = await planner.resolve(judge, DumpTransport(), "What is the battery temperature?",
                                   executor=executor, store=RecipeStore(tmp_path))
    assert result.tier == 2 and result.status == "resolved"
    assert ran == ["dumpsys"]  # exactly once -- not twelve times
    assert len(judge.payloads) == 0  # the screen-only done-check never re-asked
    assert result.steps[0].status == "verified"


async def test_tier2_unverified_read_does_not_short_circuit(journal_recorder, tmp_path) -> None:
    """The read short-circuit keys on VERIFIED only: an unconfirmed read (no
    answer field resolved) keeps the loop fail-closed to the step bound."""
    ran: list[str] = []

    async def executor(jev, transport, kind, step_goal, **kw):
        ran.append(kind)
        return planner.StepResult(kind, step_goal, "none")

    payloads: list[dict] = []
    for _ in range(planner.MAX_PLANNER_STEPS):
        payloads.append({"satisfied": NoulAnswer(noul=0.1)})
        payloads.append(_kind_pick_payload(kind="dumpsys"))
    judge = FakeJudge(payloads)
    result = await planner.resolve(judge, DumpTransport(), "What is the battery temperature?",
                                   executor=executor, store=RecipeStore(tmp_path))
    assert result.tier == 3 and result.status == "cold_goal"
    assert ran == ["dumpsys"] * planner.MAX_PLANNER_STEPS


async def test_tier3_cold_goal_is_the_calling_agents_job(journal_recorder, tmp_path) -> None:
    async def executor(jev, transport, kind, step_goal, **kw):  # pragma: no cover
        raise AssertionError("a cold goal must never execute through the planner")

    abstain = _kind_pick_payload(kind=None)
    judge = FakeJudge([
        {"satisfied": NoulAnswer(noul=0.1)}, abstain,
        {"satisfied": NoulAnswer(noul=0.1)}, abstain,
    ])
    result = await planner.resolve(judge, DumpTransport(), "do something entirely new",
                                   executor=executor, store=RecipeStore(tmp_path))
    assert result.tier == 3 and result.status == "cold_goal"
    assert result.tier_path == [0, 1, 2, 3]
    planner_statuses = [r["status"] for r in journal_recorder.outcomes if r["status"] and r["status"].startswith("planner")]
    assert planner_statuses.count("planner_fallthrough") == 3  # tiers 0, 1, 2


async def test_tier4_human_escalation_is_explicit(journal_recorder) -> None:
    result = planner.PlannerResult(tier=3, status="cold_goal", tier_path=[0, 1, 2, 3])
    planner.mark_human_escalation("abandoned goal", result)
    rows = [r for r in journal_recorder.outcomes if r["tier"] == 4]
    assert len(rows) == 1 and rows[0]["status"] == "human_escalation"
    assert rows[0]["goal_id"] == goal_id_for("abandoned goal")
    json.dumps(rows[0], default=str)  # journal rows must stay JSONL-serializable


# --- default executor: unattended, fail-closed ---------------------------------------

def _fake_handler(*, ready: CommandVariant | None, call_id: str | None = "cid-gate"):
    class FakeHandler:
        async def propose(self, jev, transport, goal):
            gate = GateResult(verdict="needs_approval", reason="r", confidence=0.9, call_id=call_id) if call_id else None
            return SimpleNamespace(ready=ready, pending=None, reasons=(), gate_result=gate)

        async def execute(self, jev, transport, goal, proposal, command, **kw):
            return 0  # swipe/keyevent/set_dnd executors return the exit code

    return FakeHandler()


async def test_run_kind_unattended_runs_a_ready_command(monkeypatch, journal_recorder) -> None:
    from jevdevice.execution import dispatch

    monkeypatch.setenv("JEV_GRAPH_EDGE", "1")
    monkeypatch.setitem(dispatch.KIND_TABLE, "swipe", _fake_handler(ready=CommandVariant("input keyevent 4", "back")))
    result = await planner.run_kind_unattended(
        SimpleNamespace(engine_name="jev"), DumpTransport(), "swipe", "go back home")
    assert result.status == "verified"
    row = [r for r in journal_recorder.outcomes if r.get("kind") == "swipe"][-1]
    assert row["verification"] == "verified" and row["executed_command"] == "input keyevent 4"
    assert row["call_id"] == "cid-gate"
    assert row["graph_edge"] == {"from_node": "com.camera", "to_node": "com.camera"}


async def test_run_kind_unattended_never_auto_approves(monkeypatch, journal_recorder) -> None:
    from jevdevice.execution import dispatch

    monkeypatch.setitem(dispatch.KIND_TABLE, "swipe", _fake_handler(ready=None, call_id=None))
    result = await planner.run_kind_unattended(
        SimpleNamespace(engine_name="jev"), DumpTransport(), "swipe", "go back home")
    assert result.status == "escalated"
    row = [r for r in journal_recorder.outcomes if r.get("kind") == "swipe"][-1]
    assert row["verification"] == "escalated"


# --- no free-form plan generation (structural guard) ----------------------------------

def test_no_free_form_generation_in_planner_modules() -> None:
    """The planner and the recipe store never construct judge questions
    directly and carry no prompt prose: all judge wordings live in the frozen
    question set, and the planner's only judge interactions are compiled
    asks plus the closed-vocabulary kind pick."""
    banned = ("Noul(", "Choice(", "Score(", "instructions=", "You are",
              "decompose the goal", "plan the steps")
    for module in (planner, recipes_mod):
        with open(module.__file__) as handle:
            text = handle.read()
        offenders = [line for line in text.splitlines() if any(token in line for token in banned)]
        assert not offenders, f"free-form generation risk in {module.__name__}:\n" + "\n".join(offenders)
        if module is planner:
            assert "question_sets.noul(" in text and "pick_kind(" in text
