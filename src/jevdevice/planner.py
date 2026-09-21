"""Tiered planner (P7.5 T2): resolve one goal through ordered tiers, falling
through on failure -- T0 recipe hit -> T1 bounded adapt -> T2 stepwise
selection -> T3 one-shot decomposition BY THE CALLING AGENT -> T4 human
escalation. Every tier transition is journaled (outcome row `tier`/`recipe_id`/
`status=planner_*`), so per-tier telemetry (T4 of this phase) is a journal
query, not a guess.

NO FREE-FORM PLAN GENERATION ANYWHERE -- enforced structurally:
- T0/T1 replay captured recipes: the recipe fixes WHICH kinds run in WHICH
  order; the judge re-grounds each step's variable slots (element/service/
  package) through the normal propose/narrow/gate path. T1's only edit is a
  bounded DROP of a step the judge says no longer fits (a compiled Noul over
  the frozen question set) -- it can never add or rewrite a step.
- T2 is SayCan-style stepwise SELECTION over the closed ACTION_KINDS
  vocabulary (dispatch.pick_kind: a real Choice with an abstain option,
  gate-checked) + the normal gated one-action execution. Language plausibility
  = the kind pick; affordance = the propose/narrow/gate fits; loop bounded by
  MAX_PLANNER_STEPS with a compiled done-check (verify.satisfied_after_action).
- T3 is deliberately NOT the planner's job: on a cold goal the planner
  journals a cold_goal row and hands the goal back to the calling agent
  (device_do per step). Capture happens for free: a successful caller-driven
  chain lands in the journal and recipes_from_journal() aggregates it into a
  recipe -- selection over what actually worked, never generation.
- T4 is the recorded give-up: mark_human_escalation journals it.

Unattended safety: the default executor NEVER auto-approves. A needs_approval
verdict is an escalated step (fail-closed) -- in unattended runs escalations
count as failures, exactly like the P5/P7 harness rule; device_approve stays
human. Everything the planner runs goes through the existing gated dispatch
(KIND_TABLE / ungated runners) -- no gate semantics are touched here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import outcomes, question_sets
from .app_launch import launch_app_for_goal
from .budget import current_profile
from .decision_log import ESCALATED, FAILED, NONE, VERIFIED, goal_id_for, goal_scope
from .dispatch import KIND_TABLE, call_id_of, pick_kind, response_for
from .elements import describe_screen, dump_screen, screen_summary
from .recipes import Recipe, RecipeStore
from .services import run_dumpsys_query, take_screenshot
from .ui import scroll_to_find

MAX_PLANNER_STEPS = 12        # T2 loop bound: one goal never costs more than this many actions
T2_ESCALATION_LIMIT = 2       # consecutive unresolvable steps (abstain/escalate) before T3


@dataclass
class StepResult:
    kind: str | None
    goal: str
    status: str          # verified | failed | none | escalated | skipped
    call_id: str | None = None
    command: str | None = None


@dataclass
class PlannerResult:
    tier: int            # the tier that resolved it (or the tier it escalated FROM)
    status: str          # resolved | cold_goal | human_escalation
    recipe_id: str | None = None
    recipe_score: float | None = None
    steps: list[StepResult] = field(default_factory=list)
    tier_path: list[int] = field(default_factory=list)  # tiers attempted, in order


# --- default executor: one gated atomic action, unattended ----------------------

_UNGATED_RUNNERS = {
    "open_app": lambda jev, transport, goal: launch_app_for_goal(jev, transport, goal, verbose=False),
    "dumpsys": lambda jev, transport, goal: run_dumpsys_query(jev, transport, goal, verbose=False),
    "scroll_to_find": lambda jev, transport, goal: scroll_to_find(jev, transport, goal, verbose=False),
    "screenshot": lambda jev, transport, goal: take_screenshot(transport),
}


async def run_kind_unattended(
    jev, transport, kind: str, goal: str, *, tier: int | None = None, recipe_id: str | None = None,
) -> StepResult:
    """One atomic action through the existing gated dispatch, with NO approval
    path (never auto-approves; needs_approval == escalated, fail-closed). The
    outcome row carries kind/tier/recipe_id + a guarded graph_edge, so a
    planner run is capturable as a recipe just like an MCP-driven one."""
    runner = _UNGATED_RUNNERS.get(kind)
    if runner is not None:
        before = await outcomes.foreground_safe(transport)
        result = await runner(jev, transport, goal)
        response = response_for(kind, result, jev.engine_name)
        call_id = getattr(result, "call_id", None)
        edge = None
        if kind in ("open_app", "scroll_to_find"):
            after = result.package if kind == "open_app" and getattr(result, "launched", False) and result.package else await outcomes.foreground_safe(transport)
            edge = {"from_node": before, "to_node": after} if (before or after) else None
        verification = outcomes.verification_from_response(response)
        outcomes.emit_outcome(
            transport=transport, call_id=call_id,
            executed_command=getattr(result, "executed", [None])[-1] if getattr(result, "executed", None) else None,
            verification=verification, response=response,
            kind=kind, tier=tier, recipe_id=recipe_id, graph_edge=edge,
        )
        return StepResult(kind, goal, _status_of(verification), call_id)

    handler = KIND_TABLE[kind]
    proposal = await handler.propose(jev, transport, goal)
    # Fail-closed: only an ALREADY-approved command runs. A pending one waits
    # for device_approve -- an unattended planner run treats it as escalated.
    command = proposal.ready
    call_id = call_id_of(proposal)
    if command is None:
        outcomes.emit_outcome(transport=transport, call_id=call_id, verification=ESCALATED,
                              status="escalated", kind=kind, tier=tier, recipe_id=recipe_id)
        return StepResult(kind, goal, "escalated", call_id)
    outcome, edge = await outcomes.graph_edge_around(
        transport, lambda: handler.execute(jev, transport, goal, proposal, command, verify=True, verbose=False),
    )
    response = response_for(kind, outcome, jev.engine_name)
    verification = outcomes.verification_from_response(response)
    outcomes.emit_outcome(
        transport=transport, call_id=call_id, executed_command=command.command,
        verification=verification, response=response,
        kind=kind, tier=tier, recipe_id=recipe_id, graph_edge=edge,
    )
    return StepResult(kind, goal, _status_of(verification), call_id, command.command)


def _status_of(verification: str) -> str:
    return {VERIFIED: "verified", FAILED: "failed"}.get(verification, "none")


# --- tier journaling ----------------------------------------------------------

def _planner_row(status: str, *, tier: int, recipe_id: str | None = None, reasons=None) -> None:
    outcomes.emit_outcome(verification=NONE, status=status, kind=None,
                          tier=tier, recipe_id=recipe_id, reasons=list(reasons or ()))


# --- tiers ---------------------------------------------------------------------

async def _replay_chain(
    jev, transport, goal: str, chain: list[tuple[str, str]], *,
    executor, tier: int, recipe_id: str | None, check_fit: bool = False,
) -> tuple[list[StepResult], StepResult | None]:
    """Run the chain in order. check_fit=True (T1) first asks the judge a
    compiled still-fits Noul per step and skips the ones that no longer apply
    -- a bounded DROP, never an addition or rewrite. Returns (results, first
    failing step or None)."""
    results: list[StepResult] = []
    profile = current_profile(jev.engine_name)
    for kind, step_goal in chain:
        if check_fit:
            truncation: dict = {}
            screen = screen_summary(await dump_screen(transport), goal=step_goal, telemetry=truncation)
            answers = await jev.ask(
                {"goal": step_goal, "screen": screen},
                {"fits": question_sets.noul("planner.step_still_fits", step_kind=kind, step_goal=step_goal)},
                phase="recall", truncation=truncation,
            )
            if answers["fits"].noul < profile.noul_floor:
                results.append(StepResult(kind, step_goal, "skipped"))
                continue
        result = await executor(jev, transport, kind, step_goal, tier=tier, recipe_id=recipe_id)
        results.append(result)
        if result.status != "verified":
            return results, result
    return results, None


async def _saycan_loop(
    jev, transport, goal: str, *, executor,
) -> tuple[list[StepResult], bool]:
    """T2: per state, select the next atomic kind over the CLOSED vocabulary
    (pick_kind -- real Choice, abstain option, gate-checked) and run it through
    the gated executor; loop until the compiled done-check confirms the goal or
    the bounds run out. Returns (results, resolved)."""
    results: list[StepResult] = []
    profile = current_profile(jev.engine_name)
    escalations = 0
    for _ in range(MAX_PLANNER_STEPS):
        truncation: dict = {}
        screen_after = describe_screen(await dump_screen(transport), goal=goal,
                                       limit=profile.screen_limit, telemetry=truncation)
        done = await jev.ask(
            {"goal": goal, "acted_on": "planner", "screen_after": screen_after},
            {"satisfied": question_sets.noul("verify.satisfied_after_action")},
            phase="verify", truncation=truncation,
        )
        if done["satisfied"].noul >= profile.noul_floor:
            return results, True
        pick = await pick_kind(jev, goal, transport, verbose=False)
        if pick.kind is None:
            escalations += 1
            results.append(StepResult(None, goal, "escalated", pick.call_id))
            if escalations >= T2_ESCALATION_LIMIT:
                return results, False
            continue
        result = await executor(jev, transport, pick.kind, goal, tier=2)
        results.append(result)
        escalations = escalations + 1 if result.status == "escalated" else 0
        if escalations >= T2_ESCALATION_LIMIT:
            return results, False
    return results, False


async def resolve(
    jev, transport, goal: str, *, executor=None, store: RecipeStore | None = None,
    verbose: bool = False,
) -> PlannerResult:
    """One goal through T0 -> T1 -> T2 -> T3 with journaled fall-throughs.
    T4 (give up to a human) is mark_human_escalation -- a separate, explicit
    act, never an automatic one."""
    executor = executor or run_kind_unattended
    store = store or RecipeStore()
    tier_path: list[int] = []
    all_steps: list[StepResult] = []
    with goal_scope(goal):
        # --- T0: recipe hit (exact goal_id, else goal-shape retrieval) ---------
        recipe: Recipe | None = store.get(goal_id_for(goal))
        score: float | None = None
        if recipe is None and (hit := store.best_match(goal)) is not None:
            recipe, score = hit
        tier_path.append(0)
        t0_remaining = 0  # how many chain steps T1 must still consider (T0's verified prefix is live on the device)
        if recipe is not None:
            chain = [(step.kind, step.goal) for step in recipe.steps]
            steps, failed = await _replay_chain(jev, transport, goal, chain, executor=executor, tier=0, recipe_id=recipe.recipe_id)
            all_steps.extend(steps)
            t0_remaining = len(steps) - (1 if failed is not None else 0)
            if failed is None:
                _planner_row("planner_resolved", tier=0, recipe_id=recipe.recipe_id)
                return PlannerResult(0, "resolved", recipe.recipe_id, score, all_steps, tier_path)
            _planner_row("planner_fallthrough", tier=0, recipe_id=recipe.recipe_id,
                         reasons=[f"step {failed.kind!r} for {failed.goal!r} ended {failed.status}"])
        else:
            _planner_row("planner_fallthrough", tier=0, reasons=["no recipe at or above the retrieval floor"])

        # --- T1: bounded adapt of the recipe T0 already retrieved --------------
        # (No recipe above the floor at T0 means T1 has nothing to adapt either
        # -- the tiers only fall THROUGH on failure, never back onto a wider net.)
        tier_path.append(1)
        if recipe is not None:
            chain = [(step.kind, step.goal) for step in recipe.steps[t0_remaining:]]
            steps, failed = await _replay_chain(jev, transport, goal, chain, executor=executor, tier=1,
                                                recipe_id=recipe.recipe_id, check_fit=True)
            all_steps.extend(steps)
            if failed is None:
                _planner_row("planner_resolved", tier=1, recipe_id=recipe.recipe_id)
                return PlannerResult(1, "resolved", recipe.recipe_id, score, all_steps, tier_path)
            _planner_row("planner_fallthrough", tier=1, recipe_id=recipe.recipe_id,
                         reasons=[f"adapted step {failed.kind!r} for {failed.goal!r} ended {failed.status}"])
        else:
            _planner_row("planner_fallthrough", tier=1, reasons=["no recipe to adapt"])

        # --- T2: stepwise selection over the closed verb vocabulary ------------
        tier_path.append(2)
        steps, resolved = await _saycan_loop(jev, transport, goal, executor=executor)
        all_steps.extend(steps)
        if resolved:
            _planner_row("planner_resolved", tier=2)
            return PlannerResult(2, "resolved", None, None, all_steps, tier_path)
        _planner_row("planner_fallthrough", tier=2, reasons=["stepwise selection exhausted or kept escalating"])

        # --- T3: cold goal -- the CALLING AGENT decomposes (never this code) ---
        tier_path.append(3)
        _planner_row("cold_goal", tier=3, reasons=["handing the goal to the calling agent: one device_do per step, each verify-gated; a verified chain is captured as a recipe by journal aggregation"])
        return PlannerResult(3, "cold_goal", recipe.recipe_id if recipe is not None else None,
                             score, all_steps, tier_path)


def mark_human_escalation(goal: str, result: PlannerResult) -> None:
    """T4: the recorded give-up. Called explicitly by a caller that has decided
    a human must take over; capture still happens via journal aggregation if a
    human later resolves the goal through the normal flows."""
    with goal_scope(goal):
        _planner_row("human_escalation", tier=4,
                     reasons=[f"caller gave up after tiers {result.tier_path}"])
