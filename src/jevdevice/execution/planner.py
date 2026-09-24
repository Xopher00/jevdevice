"""Resolve a goal through ordered strategies, cheapest first, falling through
to the next on failure. Every transition is journaled (outcome rows carry
tier/recipe_id), so per-tier statistics are a journal query.

0. Recipe hit: run a stored verified chain as-is. The judge only re-grounds
   each step's target (element/service/package) on the live screen.
1. Recipe adapt: the nearest stored recipe minus steps the judge says no
   longer fit. Drop-only: steps are skipped, never added or rewritten.
2. Stepwise selection: repeatedly pick the next single action over the closed
   ACTION_KINDS vocabulary and run it gated, until a compiled done-check
   confirms the goal or a bound (MAX_PLANNER_STEPS / ESCALATION_LIMIT) trips.
3. Cold goal: hand the goal back to the calling agent, which decomposes it
   itself (one device_do per step). A verified chain is then captured as a
   recipe for free by journal aggregation.
4. Human escalation: an explicit, recorded give-up (mark_human_escalation).

Every judge interaction is a compiled question or a choice over a closed
vocabulary; there is no free-form plan generation anywhere here. Execution
runs through dispatch.run_kind without an approval hook, so a needs_approval
verdict counts as an escalation -- unattended runs cannot approve anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typesymbolic.domain import ActStep
from typesymbolic.engine import Episode

from jevdevice import question_sets
from jevdevice.actions.elements import describe_screen, dump_screen, screen_summary
from jevdevice.budget import current_profile
from jevdevice.jev import ask
from jevdevice.journal import decision_log, outcomes
from jevdevice.journal.decision_log import goal_id_for, goal_scope

from .dispatch import pick_kind, run_kind
from .recipes import Recipe, RecipeStore

MAX_PLANNER_STEPS = 12  # stepwise-selection bound: most actions one goal may take
ESCALATION_LIMIT = 2    # consecutive unresolvable steps before giving up on a tier
# Kinds whose device-verified result is itself the goal's deliverable: a
# dumpsys "ok" resolves a service AND an answer field against the goal from
# real command output -- output-grounded satisfaction the screen-only
# done-check cannot observe, because reads leave the screen unchanged.
# Without this recognition the stepwise loop re-runs the same verified read
# until the step bound trips and the goal falls through cold (planner batch
# evidence: "What is the battery temperature?" ran `dumpsys battery` twelve
# times, every step verified, then cold_goal). UI kinds stay out on purpose:
# their verification is action-scoped ("this tap did what it said"), not
# goal-scoped, so the loop's screen check remains the arbiter for compound
# screen-trackable goals.
OUTPUT_VERIFIED_KINDS = ("dumpsys",)


@dataclass
class PlannerResult:
    tier: int            # tier that resolved it (or escalated from)
    status: str          # resolved | cold_goal
    recipe_id: str | None = None
    recipe_score: float | None = None
    steps: list[ActStep] = field(default_factory=list)
    tier_path: list[int] = field(default_factory=list)  # tiers attempted, in order


def _step(kind: str | None, goal: str, status: str) -> ActStep:  # (kind, goal, status) onto core's ActStep
    return ActStep(name=kind or "", succeeded=None if status == "skipped" else status == "verified",
                   detail={"goal": goal, "status": status})


async def run_kind_unattended(
    jev, device, kind: str, goal: str, *, tier: int | None = None, recipe_id: str | None = None,
) -> ActStep:
    """One action through the shared dispatch, with no approval hook."""
    response = await run_kind(jev, device, kind, goal, tier=tier, recipe_id=recipe_id)
    return _step(kind, goal, outcomes.verdict_from_response(response).status)


def _planner_row(status: str, *, tier: int, recipe_id: str | None = None, reasons=None) -> None:
    outcomes.record_action(status=status, tier=tier, recipe_id=recipe_id, reasons=list(reasons or ()))


async def _replay_chain(
    jev, device, chain: list[tuple[str, str]], *,
    executor, tier: int, recipe_id: str | None, check_fit: bool = False,
) -> tuple[list[ActStep], ActStep | None]:
    """Run the chain in order. With check_fit, first ask a compiled
    still-fits question per step and skip the ones that no longer apply.
    Returns (results, first failing step or None)."""
    results: list[ActStep] = []
    floor = current_profile(jev.name).noul_floor
    for kind, step_goal in chain:
        if check_fit:
            truncation: dict = {}
            screen = screen_summary(await dump_screen(device), goal=step_goal, telemetry=truncation)
            _, answers = await ask(
                jev,
                {"goal": step_goal, "screen": screen},
                {"fits": question_sets.noul("planner.step_still_fits", step_kind=kind, step_goal=step_goal)},
                phase="recall", truncation=truncation,
            )
            if answers["fits"].noul < floor:
                results.append(_step(kind, step_goal, "skipped"))
                continue
        result = await executor(jev, device, kind, step_goal, tier=tier, recipe_id=recipe_id)
        results.append(result)
        if not result.succeeded:
            return results, result
    return results, None


async def _stepwise_loop(jev, device, goal: str, *, executor) -> tuple[list[ActStep], bool]:
    """Pick the next action over the closed vocabulary, run it gated, repeat
    until the done-check confirms the goal or a bound trips."""
    results: list[ActStep] = []
    profile = current_profile(jev.name)
    escalations = 0
    for _ in range(MAX_PLANNER_STEPS):
        truncation: dict = {}
        screen_after = describe_screen(await dump_screen(device), goal=goal,
                                      limit=profile.screen_limit, telemetry=truncation)
        _, done = await ask(
            jev,
            {"goal": goal, "acted_on": "planner", "screen_after": screen_after},
            {"satisfied": question_sets.noul("verify.satisfied_after_action")},
            phase="verify", truncation=truncation,
        )
        if done["satisfied"].noul >= profile.noul_floor:
            return results, True
        pick = await pick_kind(jev, goal, device, verbose=False)
        if pick.kind is None:
            escalations += 1
            results.append(_step(None, goal, "escalated"))
            if escalations >= ESCALATION_LIMIT:
                return results, False
            continue
        result = await executor(jev, device, pick.kind, goal, tier=2)
        results.append(result)
        if result.succeeded and result.name in OUTPUT_VERIFIED_KINDS:
            # The verified read answered the goal from device output; the
            # screen-only done-check cannot observe it, so recognize it here
            # instead of looping on an unchanged screen.
            return results, True
        escalations = escalations + 1 if result.detail.get("status") == "escalated" else 0
        if escalations >= ESCALATION_LIMIT:
            return results, False
    return results, False


async def resolve(
    jev, device, goal: str, *, executor=None, store: RecipeStore | None = None,
) -> PlannerResult:
    """One goal through the tiers, cheapest first, with journaled
    fall-throughs. Giving up to a human is mark_human_escalation -- a
    separate, explicit act, never an automatic one."""
    executor = executor or run_kind_unattended
    store = store or RecipeStore()
    tier_path: list[int] = []
    all_steps: list[ActStep] = []
    with goal_scope(goal), outcomes.episode_scope(Episode(decision_log.get_journal()).episode_id):
        # --- tier 0: recipe hit (exact goal id, else text-similarity match) --
        recipe: Recipe | None = store.get(goal_id_for(goal))
        score: float | None = None
        if recipe is None and (hit := store.best_match(goal)) is not None:
            recipe, score = hit
        tier_path.append(0)
        adapt_from = 0  # chain index tier 1 resumes from: tier 0's verified prefix is done
        if recipe is not None:
            chain = [(step.kind, step.goal) for step in recipe.steps]
            steps, failed = await _replay_chain(jev, device, chain, executor=executor, tier=0, recipe_id=recipe.recipe_id)
            all_steps.extend(steps)
            adapt_from = len(steps) - (1 if failed is not None else 0)
            if failed is None:
                _planner_row("planner_resolved", tier=0, recipe_id=recipe.recipe_id)
                return PlannerResult(0, "resolved", recipe.recipe_id, score, all_steps, tier_path)
            d = failed.detail
            _planner_row("planner_fallthrough", tier=0, recipe_id=recipe.recipe_id,
                         reasons=[f"step {failed.name!r} for {d.get('goal')!r} ended {d.get('status')}"])
        else:
            _planner_row("planner_fallthrough", tier=0, reasons=["no recipe at or above the retrieval floor"])

        # --- tier 1: adapt the recipe tier 0 already retrieved (drop-only) --
        # No recipe above the floor at tier 0 means nothing to adapt here
        # either: tiers fall through on failure, never onto a wider net.
        tier_path.append(1)
        if recipe is not None:
            chain = [(step.kind, step.goal) for step in recipe.steps[adapt_from:]]
            steps, failed = await _replay_chain(jev, device, chain, executor=executor, tier=1,
                                                recipe_id=recipe.recipe_id, check_fit=True)
            all_steps.extend(steps)
            if failed is None:
                _planner_row("planner_resolved", tier=1, recipe_id=recipe.recipe_id)
                return PlannerResult(1, "resolved", recipe.recipe_id, score, all_steps, tier_path)
            d = failed.detail
            _planner_row("planner_fallthrough", tier=1, recipe_id=recipe.recipe_id,
                         reasons=[f"adapted {failed.name!r} for {d.get('goal')!r} ended {d.get('status')}"])
        else:
            _planner_row("planner_fallthrough", tier=1, reasons=["no recipe to adapt"])

        # --- tier 2: stepwise selection over the closed vocabulary -----------
        tier_path.append(2)
        steps, resolved = await _stepwise_loop(jev, device, goal, executor=executor)
        all_steps.extend(steps)
        if resolved:
            _planner_row("planner_resolved", tier=2)
            return PlannerResult(2, "resolved", None, None, all_steps, tier_path)
        _planner_row("planner_fallthrough", tier=2, reasons=["stepwise selection exhausted or kept escalating"])

        # --- tier 3: cold goal -- the calling agent decomposes, not this code
        tier_path.append(3)
        _planner_row("cold_goal", tier=3,
                     reasons=["handing the goal to the calling agent: one device_do per step, each verify-gated; a verified chain is captured as a recipe by journal aggregation"])
        return PlannerResult(3, "cold_goal", recipe.recipe_id if recipe is not None else None,
                             score, all_steps, tier_path)


def mark_human_escalation(goal: str, result: PlannerResult) -> None:
    """The recorded give-up: called by a caller that has decided a human must
    take over. Capture still happens if a human later resolves the goal."""
    with goal_scope(goal):
        _planner_row("human_escalation", tier=4,
                     reasons=[f"caller gave up after tiers {result.tier_path}"])
