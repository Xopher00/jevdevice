"""Which ONE atomic action a goal wants, and how to run it: Jev's Choice over ACTION_KINDS
picks the kind; KIND_TABLE normalizes each kind's real propose/execute into one shape, so
run_toolkit (CLI) and mcp_server.py's device_do/device_approve (MCP) share one dispatch
instead of independently-maintained switches that can silently drift apart. Sequencing
multi-step goals is always the caller's job -- this only ever resolves one goal to one kind.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from jevdevice import question_sets
from jevdevice.actions.app_launch import launch_app_for_goal
from jevdevice.actions.elements import dump_screen, screen_summary
from jevdevice.actions.services import (
    DND_MODES,
    TOGGLEABLE_SERVICES,
    execute_command,
    execute_toggle,
    propose_dnd,
    propose_keyevent,
    propose_toggle,
    run_dumpsys_query,
    take_screenshot,
)
from jevdevice.actions.ui import (
    execute_tap,
    execute_type,
    propose_long_press,
    propose_swipe,
    propose_tap,
    propose_type,
    scroll_to_find,
)
from jevdevice.budget import choice_criteria, current_profile, is_abstain
from jevdevice.common import bootstrap, gated
from jevdevice.device import Device
from jevdevice.jev import JudgeEngine, ask
from jevdevice.journal import outcomes
from jevdevice.journal.decision_log import ESCALATED, goal_scope
from jevdevice.judge.gate import CommandVariant, confirm_with_human

# pick_kind() picks exactly one of these per goal; sequencing multi-step goals is the caller's job.
ACTION_KINDS = {
    "open_app": "finds and opens/launches an installed app for the goal",
    "dumpsys": "reads a live system service's status",
    "toggle_service": f"turns one of these radios/services on or off: {', '.join(TOGGLEABLE_SERVICES)}",
    "tap": "taps a real on-screen clickable element",
    "long_press": "long-presses a real on-screen element to open its options",
    "type_text": "types real text into a real on-screen editable field",
    "keyevent": "presses a hardware/software key (home, back, volume, media, camera)",
    "swipe": "swipes/scrolls the screen up, down, left, or right",
    "scroll_to_find": "scrolls repeatedly until a specific real on-screen item is visible",
    "screenshot": "saves a screenshot of the current screen",
    "set_dnd": f"sets Do Not Disturb to one of: {', '.join(DND_MODES)}",
}


@dataclass
class KindPick:
    kind: str | None
    confidence: float
    reasons: tuple[str, ...] = ()
    call_id: str | None = None  # journal linkage: escalated kind picks join their outcome row via this


async def pick_kind(jev: JudgeEngine, goal: str, device: Device | None = None, *, verbose: bool = True) -> KindPick:
    """Which ONE atomic action kind this goal is asking for -- the calling agent
    (human at the CLI, or an LLM composing MCP tool calls) decides sequencing;
    this only ever resolves a single goal to a single kind. `transport`, when given,
    grounds the pick in what's really on screen -- without it, an ambiguous goal like
    "search for X" can't be told apart from "open an app named X"."""
    if verbose:
        print("--- Jev picks the action kind (real Choice over ACTION_KINDS) ---")
    state: dict = {"goal": goal, "action_options": ACTION_KINDS}
    # The screen-grounded variant is its own frozen entry
    # (kind.pick_screen) -- nothing is composed at runtime.
    kind_question_id = "kind.pick"
    truncation: dict = {}
    if device is not None:
        try:
            summary = screen_summary(await dump_screen(device), goal=goal, telemetry=truncation)
            # on_screen's full label list dilutes confidence even on an unrelated
            # goal -- not worth it for kind selection.
            state["screen"] = {"foreground_package": summary["foreground_package"], "editable_fields": summary["editable_fields"]}
            kind_question_id = "kind.pick_screen"
        except Exception:  # noqa: BLE001, S110 -- any dump failure just means picking the kind without screen grounding
            pass
    call_id, answers = await ask(
        jev,
        state,
        {
            "kind": question_sets.choice(kind_question_id, choice_criteria(ACTION_KINDS, current_profile(jev.name))),
            "any_fit": question_sets.noul("kind.any_fit"),
        },
        phase="kind", truncation=truncation,
    )
    kind_pick = answers["kind"]
    if verbose:
        print(f"picked kind: {kind_pick.choice} (confidence {kind_pick.confidence:.2f}, any_fit {answers['any_fit'].noul:.2f})\n")
    if is_abstain(kind_pick.choice):
        return KindPick(None, kind_pick.confidence, ("judge abstained: picked none_of_these, so no action kind fits this goal",), call_id=call_id)
    profile = current_profile(jev.name)
    if answers["any_fit"].noul < profile.noul_floor:
        return KindPick(None, kind_pick.confidence, (f"none of the {len(ACTION_KINDS)} action kinds fit this goal",), call_id=call_id)
    kind = gated(kind_pick, profile=profile)
    if kind is None:
        return KindPick(None, kind_pick.confidence, ("confidence gate rejected the action-kind pick",), call_id=call_id)
    return KindPick(kind, kind_pick.confidence, call_id=call_id)


@dataclass
class KindHandler:
    """Normalizes one kind's real propose/execute signatures into one shape -- the single
    source run_toolkit (CLI) and mcp_server.py's device_do/device_approve (MCP) both dispatch
    through, so ACTION_KINDS and the actual dispatch can no longer silently drift apart."""
    propose: Callable[[JudgeEngine, Device, str], Awaitable]
    execute: Callable[..., Awaitable]
    resume_arg: Callable[[object], object]


KIND_TABLE: dict[str, KindHandler] = {
    "toggle_service": KindHandler(
        propose=lambda jev, transport, goal: propose_toggle(jev, transport, goal, verbose=False),
        execute=lambda jev, transport, goal, proposal, command, **kw: execute_toggle(jev, transport, goal, proposal.service, command, verbose=False),
        resume_arg=lambda proposal: proposal.service,
    ),
    "tap": KindHandler(
        propose=lambda jev, transport, goal: propose_tap(jev, transport, goal, verbose=False),
        execute=lambda jev, transport, goal, proposal, command, *, verify=True, **kw: execute_tap(jev, transport, goal, proposal.element, command, confidence=proposal.confidence, verify=verify, verbose=False),
        resume_arg=lambda proposal: proposal.element,
    ),
    "long_press": KindHandler(
        propose=lambda jev, transport, goal: propose_long_press(jev, transport, goal, verbose=False),
        execute=lambda jev, transport, goal, proposal, command, *, verify=True, **kw: execute_tap(jev, transport, goal, proposal.element, command, confidence=proposal.confidence, verify=verify, verbose=False),
        resume_arg=lambda proposal: proposal.element,
    ),
    "type_text": KindHandler(
        propose=lambda jev, transport, goal: propose_type(jev, transport, goal, verbose=False),
        execute=lambda jev, transport, goal, proposal, command, *, verify=True, **kw: execute_type(jev, transport, goal, proposal.element, command, confidence=proposal.confidence, verify=verify, verbose=False),
        resume_arg=lambda proposal: proposal.element,
    ),
    "keyevent": KindHandler(
        propose=lambda jev, transport, goal: propose_keyevent(jev, goal, verbose=False),
        execute=lambda jev, transport, goal, proposal, command, **kw: execute_command(transport, command),
        resume_arg=lambda proposal: None,
    ),
    "swipe": KindHandler(
        propose=lambda jev, transport, goal: propose_swipe(jev, transport, goal, verbose=False),
        execute=lambda jev, transport, goal, proposal, command, **kw: execute_command(transport, command),
        resume_arg=lambda proposal: None,
    ),
    "set_dnd": KindHandler(
        propose=lambda jev, transport, goal: propose_dnd(jev, goal, verbose=False),
        execute=lambda jev, transport, goal, proposal, command, **kw: execute_command(transport, command),
        resume_arg=lambda proposal: None,
    ),
}


def _resolve_pending(proposal, *, verbose: bool) -> CommandVariant | None:
    """CLI-only: resolve a Pending via a blocking prompt. Returns the command
    to run, or None if it was never approved."""
    if proposal.ready is not None:
        return proposal.ready
    command = None
    if proposal.pending is not None:
        approved = confirm_with_human(proposal.pending.command, proposal.pending.chosen_label, proposal.pending.gate_result.confidence)
        command = proposal.pending.command if approved else None
    if command is None and verbose:
        print(f"=== NOT EXECUTED === {'; '.join(proposal.reasons) or 'gate did not approve'}")
    return command


async def run_toolkit(jev: JudgeEngine, device: Device, goal: str, *, verbose: bool = True):
    """CLI convenience: one goal -> one atomic action, resolving needs_approval
    with a blocking prompt. Sequencing multi-step goals is the caller's job --
    run this (or the matching MCP tool) once per step."""
    with goal_scope(goal):
        return await _run_toolkit_scoped(jev, device, goal, verbose=verbose)


async def _run_toolkit_scoped(jev: JudgeEngine, device: Device, goal: str, *, verbose: bool = True):
    if verbose:
        print(f"goal: {goal!r}\n")
    pick = await pick_kind(jev, goal, device, verbose=verbose)
    if pick.kind is None:
        if verbose:
            print(f"=== ESCALATED === {'; '.join(pick.reasons)}")
        return None

    if pick.kind == "open_app":
        return await launch_app_for_goal(jev, device, goal, verbose=verbose)
    if pick.kind == "dumpsys":
        return await run_dumpsys_query(jev, device, goal, verbose=verbose)
    if pick.kind == "scroll_to_find":
        return await scroll_to_find(jev, device, goal, verbose=verbose)
    if pick.kind == "screenshot":
        png = await take_screenshot(device)
        if verbose:
            print(f"screenshot: {len(png)} bytes (CLI doesn't display images -- pull it from the caller if needed)")
        return png

    handler = KIND_TABLE.get(pick.kind)
    if handler is None:
        if verbose:
            print(f"(no dispatch wired for {pick.kind!r})")
        return None
    proposal = await handler.propose(jev, device, goal)
    command = _resolve_pending(proposal, verbose=verbose)
    if command is None:
        return None
    return await handler.execute(jev, device, goal, proposal, command, verify=True, verbose=verbose)


# --- shared response builders ------------------------------------------------
# The dict-response half of the dispatch: one builder per kind's outcome, so
# every caller renders the same outcome identically. Only the gate-profile
# comparisons need the engine; the rest ignore it.


def _toggle_response(outcome, engine_name: str) -> dict:
    return {
        # Weakest-link status: a clean resolve with a low post-verify `satisfied` still isn't "ok".
        "status": "ok" if not outcome.reasons and outcome.satisfied >= current_profile(engine_name).noul_floor else "unverified",
        "exit_code": outcome.exit_code,
        "check_service": outcome.check_service,
        "satisfied": outcome.satisfied,
        "reasons": list(outcome.reasons),
    }


def _tap_response(outcome, engine_name: str) -> dict:
    return {
        "status": "ok" if outcome.tapped and outcome.satisfied >= current_profile(engine_name).noul_floor else "unverified",
        "element": outcome.element,
        "satisfied": outcome.satisfied,
        "reasons": list(outcome.reasons),
    }


def _launch_response(outcome, engine_name: str) -> dict:
    return {
        "status": "ok" if outcome.launched else "escalated",
        "package": outcome.package,
        "satisfied": outcome.satisfied,
        "reasons": list(outcome.reasons),
    }


def _dumpsys_response(outcome, engine_name: str) -> dict:
    # Weakest-link status: resolving the service but not the answer field is real, not "ok".
    if outcome.parsed is None:
        status = "escalated"
    elif outcome.answer_key is None:
        status = "unverified"
    else:
        status = "ok"
    return {
        "status": status,
        "service": outcome.service,
        "answer": {outcome.answer_key: outcome.parsed[outcome.answer_key]} if outcome.answer_key else None,
        "parsed": outcome.parsed,
        "reasons": list(outcome.reasons),
    }


def _exit_code_response(outcome, engine_name: str) -> dict:
    return {"status": "ok" if outcome == 0 else "unverified", "exit_code": outcome}


def _scroll_response(outcome, engine_name: str) -> dict:
    return {
        "status": "ok" if outcome.found else "escalated",
        "found": outcome.found,
        "attempts": outcome.attempts,
        "reasons": list(outcome.reasons),
    }


def _screenshot_response(outcome, engine_name: str) -> dict:
    return {"status": "ok"}


# Kinds whose own result IS the requested artifact -- device_do attaches it
# even when the caller didn't ask for include_screenshot.
ATTACHES_RESULT = {"screenshot"}


RESPONSE_FOR = {
    "toggle_service": _toggle_response,
    "tap": _tap_response,
    "long_press": _tap_response,
    "type_text": _tap_response,
    "keyevent": _exit_code_response,
    "swipe": _exit_code_response,
    "set_dnd": _exit_code_response,
    "open_app": _launch_response,
    "dumpsys": _dumpsys_response,
    "scroll_to_find": _scroll_response,
    "screenshot": _screenshot_response,
}


def response_for(kind: str, outcome, engine_name: str) -> dict:
    """One response dict per kind's outcome -- the single table every
    caller (MCP server, planner) renders through."""
    return RESPONSE_FOR[kind](outcome, engine_name)


def call_id_of(proposal) -> str | None:
    """The gate ask's call_id carried by any proposal shape, so executed actions
    join their outcome row to the decision row that approved them."""
    gate = getattr(proposal, "gate_result", None)
    if gate is not None and gate.call_id:
        return gate.call_id
    pending = getattr(proposal, "pending", None)
    if pending is not None and pending.gate_result is not None:
        return pending.gate_result.call_id
    return None


# --- one atomic action, one shared execution path -----------------------------
# Both real callers (the MCP server and the planner) run actions through
# run_kind: propose -> gate -> execute, bracketed with a graph_edge, outcome
# row journaled with the kind that ran. The only caller-specific part is what
# happens to a needs_approval verdict, passed as `on_pending`.


async def _run_ungated(
    jev, device, kind: str, goal: str, *, direction: str, max_attempts: int,
    tier: int | None, recipe_id: str | None,
) -> dict:
    """The read-side kinds: no command gate (nothing mutates), but the same
    outcome-row journaling so trajectories join by call_id like every other
    kind. open_app/scroll_to_find change the screen, so they carry a
    graph_edge; dumpsys/screenshot are read-only and carry none."""
    if kind == "open_app":
        before = await outcomes.foreground_safe(device)
        result = await launch_app_for_goal(jev, device, goal, verbose=False)
        response = response_for(kind, result, jev.name)
        # The launch verification already proved the foreground package; ride
        # that device truth instead of paying a second dump when it ran.
        after = result.package if result.launched and result.package else await outcomes.foreground_safe(device)
        edge = {"from_node": before, "to_node": after} if (before or after) else None
        outcomes.emit_outcome(
            device=device, call_id=result.call_id,
            executed_command=f"monkey -p {result.package} 1" if result.package else None,
            verification=outcomes.verification_from_response(response), response=response,
            kind=kind, graph_edge=edge, tier=tier, recipe_id=recipe_id,
        )
        return response
    if kind == "dumpsys":
        result = await run_dumpsys_query(jev, device, goal, verbose=False)
        response = response_for(kind, result, jev.name)
        outcomes.emit_outcome(
            device=device, call_id=result.call_id,
            executed_command=f"dumpsys {result.service}" if result.service else None,
            verification=outcomes.verification_from_response(response), response=response,
            kind=kind, tier=tier, recipe_id=recipe_id,
        )
        return response
    if kind == "scroll_to_find":
        before = await outcomes.foreground_safe(device)
        result = await scroll_to_find(jev, device, goal, direction=direction, max_attempts=max_attempts, verbose=False)
        response = response_for(kind, result, jev.name)
        after = await outcomes.foreground_safe(device)
        edge = {"from_node": before, "to_node": after} if (before or after) else None
        outcomes.emit_outcome(
            device=device, call_id=result.call_id,
            executed_command=result.executed[-1] if result.executed else None,
            verification=outcomes.verification_from_response(response), response=response,
            executed=list(result.executed), kind=kind, graph_edge=edge, tier=tier, recipe_id=recipe_id,
        )
        return {**response, "executed": list(result.executed)}
    if kind == "screenshot":
        response = {"status": "ok"}
        outcomes.emit_outcome(
            device=device, call_id=None, executed_command="screencap -p",
            verification=outcomes.verification_from_response(response), response=response,
            kind=kind, tier=tier, recipe_id=recipe_id,
        )
        return response
    raise KeyError(f"not an ungated kind: {kind!r}")


async def run_kind(
    jev, device, kind: str, goal: str, *, verify: bool = True, auto_approve: bool = False,
    direction: str = "down", max_attempts: int = 8,
    on_pending: Callable[..., Awaitable[dict] | dict] | None = None,
    tier: int | None = None, recipe_id: str | None = None,
) -> dict:
    """Run exactly ONE atomic action of one kind end to end: propose -> gate
    -> execute (bracketed with a graph_edge) -> journal the outcome row with
    the kind that ran. Returns the response dict.

    A needs_approval verdict never executes here. `on_pending` receives
    (goal, kind, resume_arg, confidence, pending, verify) and returns the
    response to surface (the MCP server stores the pending action for
    device_approve); with no hook the verdict is returned as an escalation.
    `auto_approve` decides a needs_approval verdict for this call only -- a
    DENIED command still stops the action, same invariant as everywhere else.
    tier/recipe_id stamp planner-driven rows (None on ordinary rows)."""
    handler = KIND_TABLE.get(kind)
    if handler is None:
        return await _run_ungated(jev, device, kind, goal, direction=direction,
                                  max_attempts=max_attempts, tier=tier, recipe_id=recipe_id)
    t0 = time.monotonic()
    proposal = await handler.propose(jev, device, goal)
    call_id = call_id_of(proposal)
    command = proposal.ready
    if command is None and proposal.pending is not None:
        if auto_approve:
            command = proposal.pending.command
        elif on_pending is not None:
            outcomes.emit_outcome(device=device, call_id=call_id, verification=ESCALATED,
                                  status="needs_approval", kind=kind, tier=tier, recipe_id=recipe_id)
            # The hook contract (type hint above) admits sync and async alike:
            # the MCP server parks a pending action synchronously, so the
            # hook's return is awaited only when it is actually awaitable.
            hooked = on_pending(goal, kind, handler.resume_arg(proposal),
                                getattr(proposal, "confidence", 0.0), proposal.pending, verify)
            return await hooked if inspect.isawaitable(hooked) else hooked
    if command is None:
        outcomes.emit_outcome(device=device, call_id=call_id, verification=ESCALATED,
                              status="escalated", kind=kind, tier=tier, recipe_id=recipe_id)
        return {"status": "escalated", "reasons": list(proposal.reasons)}
    outcome, edge = await outcomes.graph_edge_around(
        device, lambda: handler.execute(jev, device, goal, proposal, command, verify=verify, verbose=False),
    )
    response = response_for(kind, outcome, jev.name)
    outcomes.emit_outcome(
        device=device, call_id=call_id, executed_command=command.command,
        verification=outcomes.verification_from_response(response), response=response,
        kind=kind, graph_edge=edge, tier=tier, recipe_id=recipe_id,
    )
    if kind in ("tap", "long_press", "type_text"):
        response = {**response, "elapsed_s": round(time.monotonic() - t0, 2)}
    return response


async def main() -> None:
    jev, device = bootstrap()
    goal = sys.argv[1] if len(sys.argv) > 1 else "what is my battery level?"
    await run_toolkit(jev, device, goal)
    print(f"\nusage: {jev.usage.snapshot()}")

if __name__ == "__main__":
    asyncio.run(main())
