"""Which ONE atomic action a goal wants, and how to run it: Jev's Choice over ACTION_KINDS
picks the kind; KIND_TABLE normalizes each kind's real propose/execute into one shape, so
run_toolkit (CLI) and mcp_server.py's device_do/device_approve (MCP) share one dispatch
instead of independently-maintained switches that can silently drift apart. Sequencing
multi-step goals is always the caller's job -- this only ever resolves one goal to one kind.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .app_launch import launch_app_for_goal
from .common import bootstrap, gated
from .decision_log import goal_scope
from .elements import dump_screen, screen_summary
from .gate import CommandVariant, confirm_with_human
from .jev import Choice, JevClient, Noul
from .services import (
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
from .transport import AdbTransport
from .ui import (
    execute_tap,
    execute_type,
    propose_long_press,
    propose_swipe,
    propose_tap,
    propose_type,
    scroll_to_find,
)

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


async def pick_kind(jev: JevClient, goal: str, transport: AdbTransport | None = None, *, verbose: bool = True) -> KindPick:
    """Which ONE atomic action kind this goal is asking for -- the calling agent
    (human at the CLI, or an LLM composing MCP tool calls) decides sequencing;
    this only ever resolves a single goal to a single kind. `transport`, when given,
    grounds the pick in what's really on screen (confirmed live: without it, an
    ambiguous goal like "search for X" can't be told apart from "open an app named X")."""
    if verbose:
        print("--- Jev picks the action kind (real Choice over ACTION_KINDS) ---")
    state: dict = {"goal": goal, "action_options": ACTION_KINDS}
    kind_instructions = "Which ONE action would this goal have you perform?"
    truncation: dict = {}
    if transport is not None:
        try:
            summary = screen_summary(await dump_screen(transport), goal=goal, telemetry=truncation)
            # on_screen's full label list measurably dilutes confidence even on an unrelated
            # goal (confirmed live: 1.00 -> 0.47-0.69) -- not worth it for kind selection.
            state["screen"] = {"foreground_package": summary["foreground_package"], "editable_fields": summary["editable_fields"]}
            kind_instructions += (
                " screen describes the real device right now: screen.editable_fields lists "
                "real text fields actually on screen, screen.foreground_package the app in "
                "front. A goal naming typing or searching, when a real editable field is "
                "already on screen, is type_text, not open_app."
            )
        except Exception:  # noqa: BLE001, S110 -- any dump failure just means picking the kind without screen grounding
            pass
    call_id = str(uuid.uuid4())
    answers = await jev.ask(
        state,
        {
            "kind": Choice(instructions=kind_instructions, criteria=ACTION_KINDS),
            "any_fit": Noul(instructions="Given action_options, does any of them fit this goal?"),
        },
        phase="kind", call_id=call_id, truncation=truncation,
    )
    kind_pick = answers["kind"]
    if verbose:
        print(f"picked kind: {kind_pick.choice} (confidence {kind_pick.confidence:.2f}, any_fit {answers['any_fit'].noul:.2f})\n")
    if answers["any_fit"].noul < 0.5:
        return KindPick(None, kind_pick.confidence, (f"none of the {len(ACTION_KINDS)} action kinds fit this goal",), call_id=call_id)
    kind = gated(kind_pick)
    if kind is None:
        return KindPick(None, kind_pick.confidence, ("confidence gate rejected the action-kind pick",), call_id=call_id)
    return KindPick(kind, kind_pick.confidence, call_id=call_id)


@dataclass
class KindHandler:
    """Normalizes one kind's real propose/execute signatures into one shape -- the single
    source run_toolkit (CLI) and mcp_server.py's device_do/device_approve (MCP) both dispatch
    through, so ACTION_KINDS and the actual dispatch can no longer silently drift apart."""
    propose: Callable[[JevClient, AdbTransport, str], Awaitable]
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
        approved = confirm_with_human(proposal.pending.command, proposal.pending.chosen_label, proposal.pending.gate_result.noul_confidence)
        command = proposal.pending.command if approved else None
    if command is None and verbose:
        print(f"=== NOT EXECUTED === {'; '.join(proposal.reasons) or 'gate did not approve'}")
    return command


async def run_toolkit(jev: JevClient, transport: AdbTransport, goal: str, *, verbose: bool = True):
    """CLI convenience: one goal -> one atomic action, resolving needs_approval
    with a blocking prompt. Sequencing multi-step goals is the caller's job --
    run this (or the matching MCP tool) once per step."""
    with goal_scope(goal):
        return await _run_toolkit_scoped(jev, transport, goal, verbose=verbose)


async def _run_toolkit_scoped(jev: JevClient, transport: AdbTransport, goal: str, *, verbose: bool = True):
    if verbose:
        print(f"goal: {goal!r}\n")
    pick = await pick_kind(jev, goal, transport, verbose=verbose)
    if pick.kind is None:
        if verbose:
            print(f"=== ESCALATED === {'; '.join(pick.reasons)}")
        return None

    if pick.kind == "open_app":
        return await launch_app_for_goal(jev, transport, goal, verbose=verbose)
    if pick.kind == "dumpsys":
        return await run_dumpsys_query(jev, transport, goal, verbose=verbose)
    if pick.kind == "scroll_to_find":
        return await scroll_to_find(jev, transport, goal, verbose=verbose)
    if pick.kind == "screenshot":
        png = await take_screenshot(transport)
        if verbose:
            print(f"screenshot: {len(png)} bytes (CLI doesn't display images -- pull it from the caller if needed)")
        return png

    handler = KIND_TABLE.get(pick.kind)
    if handler is None:
        if verbose:
            print(f"(no dispatch wired for {pick.kind!r})")
        return None
    proposal = await handler.propose(jev, transport, goal)
    command = _resolve_pending(proposal, verbose=verbose)
    if command is None:
        return None
    return await handler.execute(jev, transport, goal, proposal, command, verify=True, verbose=verbose)


async def main() -> None:
    jev, transport = bootstrap()
    goal = sys.argv[1] if len(sys.argv) > 1 else "what is my battery level?"
    await run_toolkit(jev, transport, goal)
    print(f"\nusage: {jev.usage.snapshot()}")


if __name__ == "__main__":
    asyncio.run(main())
