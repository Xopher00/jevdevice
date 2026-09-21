"""P7.5: recipe store + tiered planner. All unit-level: fakes for the judge,
the transport, and the per-step executor, so nothing here touches a device or
the network. The no-free-form guardrail is enforced structurally (T3 of the
phase): the planner may only ever ask the judge compiled questions or the
closed ACTION_KINDS choice -- a static scan plus behavioral asserts pin it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from jevdevice import planner
from jevdevice import recipes as recipes_mod
from jevdevice.decision_log import goal_id_for
from jevdevice.gate import CommandVariant, GateResult
from jevdevice.jev import ChoiceAnswer, NoulAnswer
from jevdevice.recipes import (
    Recipe,
    RecipeStep,
    RecipeStore,
    embed,
    recipes_from_journal,
    similarity,
    verify_recipe,
)

# --- fakes ---------------------------------------------------------------------

class RecordingJournal:
    """Duck-typed DecisionJournal: replay() yields prebuilt rows."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def replay(self):
        yield from self._rows


class LiveJournal:
    """Records emit_outcome traffic (planner rows) without touching disk."""

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


def _chain_recipe(goal: str, steps: list[RecipeStep], *, question_set: str = "v1") -> Recipe:
    return Recipe(
        recipe_id=goal_id_for(goal), goal=goal, goal_id=goal_id_for(goal), steps=steps,
        embedding=embed(goal), created="2026-09-21T00:00:00", question_set=question_set,
    )


# --- goal-shape embedding --------------------------------------------------------

def test_embed_is_deterministic_and_normalized() -> None:
    assert embed("open the clock app") == embed("open the clock app")
    vec = embed("open the clock app")
    assert abs(sum(component * component for component in vec) - 1.0) < 1e-9


def test_similar_goals_rank_above_unrelated() -> None:
    query = embed("open the camera app and take a picture")
    near = similarity(query, embed("open the camera app"))
    far = similarity(query, embed("toggle bluetooth off"))
    assert near > far
    assert far < recipes_mod.RETRIEVAL_FLOOR  # unrelated goals never match


# --- builder: SkillX compression -------------------------------------------------

def test_builder_keeps_only_the_final_verified_chain() -> None:
    goal = "set an alarm"
    rows = [
        outcome_row(goal, kind="open_app", verification="escalated", ts="2026-09-21T01:00:00"),
        outcome_row(goal, kind="tap", verification="failed", command="input tap 1 1", ts="2026-09-21T01:00:01"),
        outcome_row(goal, kind="open_app", verification="verified", command="monkey -p com.clock 1", edge={"from_node": "home", "to_node": "com.clock"}, call_id="c1", ts="2026-09-21T01:01:00"),
        outcome_row(goal, kind="tap", verification="verified", command="input tap 5 700", edge={"from_node": "com.clock", "to_node": "com.clock"}, call_id="c2", ts="2026-09-21T01:01:05"),
    ]
    built = recipes_from_journal(RecordingJournal(rows))
    recipe = built[goal_id_for(goal)]
    assert [step.kind for step in recipe.steps] == ["open_app", "tap"]
    assert [step.call_id for step in recipe.steps] == ["c1", "c2"]
    assert recipe.steps[0].to_node == "com.clock"


def test_builder_compresses_backtracking_and_dead_ends() -> None:
    goal = "toggle wifi off then on"
    rows = [
        # a verified first pass, then a backtrack (same command redone), then the final verified pass
        outcome_row(goal, kind="open_app", verification="verified", command="monkey -p com.settings 1", call_id="c1", ts="2026-09-21T02:00:00"),
        outcome_row(goal, kind="open_app", verification="verified", command="monkey -p com.settings 1", call_id="c2", ts="2026-09-21T02:00:10"),  # backtracked duplicate
        outcome_row(goal, kind="tap", verification="failed", command="input tap 1 1", call_id="c3", ts="2026-09-21T02:00:20"),  # dead end
        outcome_row(goal, kind="toggle_service", verification="verified", command="svc wifi disable", call_id="c4", ts="2026-09-21T02:00:30"),
    ]
    built = recipes_from_journal(RecordingJournal(rows))
    recipe = built[goal_id_for(goal)]
    # the earlier duplicate open_app is compressed to the last occurrence; the
    # failed tap never enters; the final verified toggle ends the chain
    assert [(step.kind, step.command) for step in recipe.steps] == [
        ("open_app", "monkey -p com.settings 1"),
        ("toggle_service", "svc wifi disable"),
    ]


def test_builder_skips_goals_without_a_verified_outcome_or_kind() -> None:
    rows = [
        outcome_row("never done", kind="tap", verification="escalated", ts="2026-09-21T03:00:00"),
        outcome_row("pre-kind rows", kind=None, verification="verified", command="old", ts="2026-09-21T03:01:00"),
    ]
    assert recipes_from_journal(RecordingJournal(rows)) == {}


def test_builder_refuses_heldout_goals() -> None:
    goal = "heldout only"
    rows = [outcome_row(goal, kind="tap", verification="verified", command="input tap 1 1", call_id="c", ts="t")]
    heldout = {goal_id_for(goal)}
    with pytest.raises(ValueError, match="held-out"):
        recipes_from_journal(RecordingJournal(rows), heldout_goal_ids=heldout)


# --- store + replay ------------------------------------------------------------

def test_store_upsert_best_match_and_persistence(tmp_path) -> None:
    recipe = _chain_recipe("open the camera app", [RecipeStep("open_app", "open the camera app", "monkey -p com.camera 1", "home", "com.camera", "c1")])
    store = RecipeStore(tmp_path)
    store.upsert(recipe)
    reloaded = RecipeStore(tmp_path)
    assert reloaded.get(goal_id_for("open the camera app")) is not None
    hit = reloaded.best_match("open the camera app and take a picture")
    assert hit is not None and hit[0].recipe_id == recipe.recipe_id
    assert reloaded.best_match("toggle bluetooth off") is None  # below the retrieval floor


def test_stored_recipe_replays_to_its_verified_outcome() -> None:
    goal = "open the camera app"
    rows = [
        outcome_row(goal, kind="open_app", verification="verified", command="monkey -p com.camera 1", call_id="c1", ts="t"),
    ]
    recipe = _chain_recipe(goal, [RecipeStep("open_app", goal, "monkey -p com.camera 1", "home", "com.camera", "c1")])
    assert verify_recipe(recipe, RecordingJournal(rows))
    drifted = _chain_recipe(goal, [RecipeStep("open_app", goal, "monkey -p com.camera 1", "home", "com.camera", "cMISSING")])
    assert not verify_recipe(drifted, RecordingJournal(rows))


# --- planner tiers ---------------------------------------------------------------

@pytest.fixture()
def journal_recorder(monkeypatch):
    from jevdevice import decision_log

    live = LiveJournal()
    monkeypatch.setattr(decision_log, "get_journal", lambda: live)
    return live


async def _tier_t0_resolves(journal_recorder, tmp_path) -> None:
    goal = "open the camera app"
    recipe = _chain_recipe(goal, [
        RecipeStep("open_app", goal, "monkey -p com.camera 1", "home", "com.camera", "c1"),
        RecipeStep("tap", goal, "input tap 5 700", "com.camera", "com.camera", "c2"),
    ])
    store = RecipeStore(tmp_path)
    store.upsert(recipe)
    ran: list[tuple[str, str]] = []

    async def executor(jev, transport, kind, step_goal, **kw):
        ran.append((kind, step_goal))
        return planner.StepResult(kind, step_goal, "verified", "cid")

    judge = FakeJudge([])
    result = await planner.resolve(judge, None, goal, executor=executor, store=store)
    assert result.tier == 0 and result.status == "resolved"
    assert ran == [("open_app", goal), ("tap", goal)]
    tiers = [row["tier"] for row in journal_recorder.outcomes if row["status"] == "planner_resolved"]
    assert tiers == [0]


def test_planner_t0_recipe_hit_resolves(journal_recorder, tmp_path) -> None:
    import asyncio

    asyncio.run(_tier_t0_resolves(journal_recorder, tmp_path))


async def _tier_t1_drops_the_stale_step(journal_recorder, tmp_path) -> None:
    goal = "open the camera app and take a picture"
    recipe = _chain_recipe(goal, [
        RecipeStep("open_app", goal, "monkey -p com.camera 1", "home", "com.camera", "c1"),
        RecipeStep("tap", "tap the shutter", "input tap 5 700", "com.camera", "com.camera", "c2"),
        RecipeStep("toggle_service", "toggle flash", "svc flash off", "com.camera", "com.camera", "c3"),
    ])
    store = RecipeStore(tmp_path)
    store.upsert(recipe)
    ran: list[str] = []

    async def executor(jev, transport, kind, step_goal, **kw):
        ran.append(kind)
        # T0: the tap (step 2) fails on the current screen; T1: everything it runs verifies
        if kind == "tap" and kw.get("tier") == 0:
            return planner.StepResult(kind, step_goal, "escalated", "cid")
        return planner.StepResult(kind, step_goal, "verified", "cid")

    # T1 asks one compiled still-fits Noul per remaining step: the stale
    # toggle_service no longer applies (bounded DROP), the tap does.
    judge = FakeJudge([
        {"fits": NoulAnswer(noul=0.9)},   # tap still fits -> run it
        {"fits": NoulAnswer(noul=0.1)},   # toggle_service no longer applies -> skipped
    ])
    result = await planner.resolve(judge, DumpTransport(), goal, executor=executor, store=store)
    assert result.tier == 1 and result.status == "resolved"
    assert ran == ["open_app", "tap", "tap"]  # T0 runs open_app+tap(escalated); T1 re-runs only the remaining prefix from the tap on
    statuses = [row["status"] for row in journal_recorder.outcomes]
    assert "planner_fallthrough" in statuses and "planner_resolved" in statuses
    skipped = [step for step in result.steps if step.status == "skipped"]
    assert [step.kind for step in skipped] == ["toggle_service"]

def test_planner_t1_bounded_adapt(journal_recorder, tmp_path) -> None:
    import asyncio

    asyncio.run(_tier_t1_drops_the_stale_step(journal_recorder, tmp_path))


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
    if kind is None:
        return {
            "kind": ChoiceAnswer(choice=NONE_OF_THESE, probabilities={NONE_OF_THESE: confidence}, confidence=confidence),
            "any_fit": NoulAnswer(noul=0.9),
        }
    return {
        "kind": ChoiceAnswer(choice=kind, probabilities={kind: confidence, NONE_OF_THESE: 0.05}, confidence=confidence),
        "any_fit": NoulAnswer(noul=0.9),
    }


async def _tier_t2_saycan(journal_recorder, tmp_path) -> None:
    store = RecipeStore(tmp_path)  # empty, isolated
    ran: list[str] = []

    async def executor(jev, transport, kind, step_goal, **kw):
        ran.append(kind)
        return planner.StepResult(kind, step_goal, "verified", "cid")

    # done-check 1: not achieved; kind pick: tap (closed vocabulary);
    # done-check 2: achieved -> resolved at T2 with ONE executed action.
    judge = FakeJudge([
        {"satisfied": NoulAnswer(noul=0.1)},
        _kind_pick_payload(),
        {"satisfied": NoulAnswer(noul=0.95)},
    ])
    result = await planner.resolve(judge, DumpTransport(), "tap the shutter button", executor=executor, store=store)
    assert result.tier == 2 and result.status == "resolved"
    assert ran == ["tap"]
    assert [row["tier"] for row in journal_recorder.outcomes if row["status"] == "planner_resolved"] == [2]


def test_planner_t2_stepwise_selection(journal_recorder, tmp_path) -> None:
    import asyncio

    asyncio.run(_tier_t2_saycan(journal_recorder, tmp_path))


async def _tier_t3_cold_goal(journal_recorder, tmp_path) -> None:
    store = RecipeStore(tmp_path)  # nothing retrievable -> T0/T1 fall through

    async def executor(jev, transport, kind, step_goal, **kw):  # pragma: no cover - never reached
        raise AssertionError("cold goal must never execute through the planner")

    abstain = _kind_pick_payload(kind=None)
    judge = FakeJudge([
        {"satisfied": NoulAnswer(noul=0.1)}, abstain,
        {"satisfied": NoulAnswer(noul=0.1)}, abstain,
    ])
    result = await planner.resolve(judge, DumpTransport(), "do something entirely new", executor=executor, store=store)
    assert result.tier == 3 and result.status == "cold_goal"
    assert result.tier_path == [0, 1, 2, 3]
    statuses = [row["status"] for row in journal_recorder.outcomes if row["status"] and row["status"].startswith("planner")]
    assert statuses.count("planner_fallthrough") == 3  # T0, T1, T2


def test_planner_t3_cold_goal_is_the_calling_agents_job(journal_recorder, tmp_path) -> None:
    import asyncio

    asyncio.run(_tier_t3_cold_goal(journal_recorder, tmp_path))


def test_planner_t4_human_escalation_is_explicit(journal_recorder) -> None:
    import asyncio

    async def main() -> None:
        goal = "abandoned goal"
        result = planner.PlannerResult(tier=3, status="cold_goal", tier_path=[0, 1, 2, 3])
        planner.mark_human_escalation(goal, result)
        rows = [row for row in journal_recorder.outcomes if row["tier"] == 4]
        assert len(rows) == 1 and rows[0]["status"] == "human_escalation"
        assert rows[0]["goal_id"] == goal_id_for(goal)

    asyncio.run(main())


# --- default executor: unattended + fail-closed ----------------------------------

def _fake_handler(monkeypatch, *, ready: CommandVariant | None, call_id="cid-gate"):
    class FakeHandler:
        async def propose(self, jev, transport, goal):
            gate = GateResult(verdict="approve", reason="r", noul_confidence=0.9, call_id=call_id) if call_id else None
            return SimpleNamespace(ready=ready, pending=None, reasons=(), gate_result=gate)

        async def execute(self, jev, transport, goal, proposal, command, **kw):
            return 0  # keyevent/swipe/set_dnd's executor returns the exit code

    return FakeHandler()


def test_run_kind_unattended_runs_a_ready_command_and_journals_graph_edge(monkeypatch, journal_recorder) -> None:
    import asyncio

    command = CommandVariant(command="input keyevent 4", rationale="back")
    monkeypatch.setitem(planner.KIND_TABLE, "swipe", _fake_handler(monkeypatch, ready=command))

    async def main() -> None:
        result = await planner.run_kind_unattended(SimpleNamespace(engine_name="jev"), DumpTransport(), "swipe", "go back home")
        assert result.status == "verified"
        row = [r for r in journal_recorder.outcomes if r.get("kind") == "swipe"][-1]
        assert row["verification"] == "verified" and row["executed_command"] == "input keyevent 4"
        assert row["call_id"] == "cid-gate"
        assert row["graph_edge"] == {"from_node": "com.camera", "to_node": "com.camera"}

    asyncio.run(main())


def test_run_kind_unattended_never_auto_approves(monkeypatch, journal_recorder) -> None:
    import asyncio

    monkeypatch.setitem(planner.KIND_TABLE, "swipe", _fake_handler(monkeypatch, ready=None, call_id=None))

    async def main() -> None:
        result = await planner.run_kind_unattended(SimpleNamespace(engine_name="jev"), DumpTransport(), "swipe", "go back home")
        assert result.status == "escalated"
        row = [r for r in journal_recorder.outcomes if r.get("kind") == "swipe"][-1]
        assert row["verification"] == "escalated"

    asyncio.run(main())


# --- no free-form plan generation (T3 guardrail) ----------------------------------

def test_no_free_form_generation_in_planner_modules() -> None:
    """Structural guard: the planner and the recipe store never construct judge
    questions directly and never carry prompt prose. All judge wordings live in
    the frozen question set; the planner's only judge interactions are compiled
    Noul/Choice asks and the closed ACTION_KINDS choice."""
    src = planner.__file__ and __import__("pathlib").Path(planner.__file__)
    rec = __import__("pathlib").Path(recipes_mod.__file__)
    banned = ("Noul(", "Choice(", "Score(", "instructions=", "You are", "decompose the goal", "plan the steps")
    for path in (src, rec):
        text = path.read_text()
        offenders = [line for line in text.splitlines() if any(token in line for token in banned)]
        assert not offenders, f"free-form generation risk in {path.name}:\n" + "\n".join(offenders)
    # the planner's judge asks all go through the frozen set or pick_kind
    text = src.read_text()
    assert "question_sets.noul(" in text and "pick_kind(" in text


def test_tier_rows_are_json_serializable(journal_recorder) -> None:
    """Every journaled tier row survives a JSON round-trip (journal is JSONL)."""
    import asyncio

    async def main() -> None:
        result = planner.PlannerResult(tier=3, status="cold_goal", tier_path=[0, 1, 2, 3],
                                       steps=[planner.StepResult(None, "g", "escalated")])
        planner.mark_human_escalation("json goal", result)
        for row in journal_recorder.outcomes:
            json.dumps(row, default=str)

    asyncio.run(main())
