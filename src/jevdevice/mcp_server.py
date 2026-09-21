"""MCP server: device_do picks and runs exactly ONE atomic action kind per call (Jev's own
Choice over ACTION_KINDS, the same one the CLI toolkit uses) -- device_screenshot and
device_approve are separate since they don't fit that shape (a real image; resuming a pending
action). The calling agent (an LLM composing these calls) still does all multi-step planning
and sequencing; device_do never chains more than one action, it only cuts the tool count the
agent has to pick from for that one action.

Each mutating action proposes + gates before running. If the gate needs a decision, device_do
returns {"status": "needs_approval", "thread_id": ...} instead of guessing; device_approve
resolves it with a second, ordinary tool call instead of a blocking prompt a tool call can't make.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image

from . import decision_log
from .app_launch import LaunchOutcome, launch_app_for_goal
from .budget import current_profile
from .common import bootstrap
from .decision_log import ESCALATED, FAILED, NONE, VERIFIED, goal_scope
from .dispatch import KIND_TABLE, pick_kind
from .elements import dump_screen, foreground_package
from .gate import CommandVariant, Pending
from .services import DumpsysOutcome, ToggleOutcome, run_dumpsys_query, take_screenshot
from .ui import ActionOutcome, scroll_to_find

mcp = FastMCP("jevdevice")
jev, transport = bootstrap()


@dataclass
class PendingAction:
    goal: str
    kind: str  # one of dispatch.KIND_TABLE's keys
    resume_arg: str | None  # a toggle's service name, or a tap/type's target element
    confidence: float
    pending: Pending
    verify: bool = True
    call_id: str | None = None  # journal linkage: joins the outcome row to the gate's decision row


_PENDING: dict[str, PendingAction] = {}


def _pending_response(action_id: str, action: PendingAction) -> dict:
    return {
        "status": "needs_approval",
        "thread_id": action_id,
        "command": action.pending.command.command,
        "rationale": action.pending.command.rationale,
        "confidence": action.pending.gate_result.noul_confidence,
    }


def _store_pending(goal: str, kind: str, resume_arg: str | None, confidence: float, pending: Pending, verify: bool = True) -> dict:
    action_id = str(uuid.uuid4())
    action = PendingAction(
        goal, kind, resume_arg, confidence, pending, verify,
        call_id=pending.gate_result.call_id if pending.gate_result else None,
    )
    _PENDING[action_id] = action
    return _pending_response(action_id, action)


# --- outcome-row emission ---------------------------------------------------

def _call_id_of(proposal) -> str | None:
    """The gate ask's call_id carried by any proposal shape, so executed actions
    join their outcome row to the decision row that approved them."""
    gate = getattr(proposal, "gate_result", None)
    if gate is not None and gate.call_id:
        return gate.call_id
    pending = getattr(proposal, "pending", None)
    if pending is not None and pending.gate_result is not None:
        return pending.gate_result.call_id
    return None


def _verification_from_response(response: dict) -> str:
    """The flow's own status -> journal verification. ok = verified; a non-zero
    exit is an outright failure; anything else that ran is unconfirmed (none)."""
    status = response.get("status")
    if status == "ok":
        return VERIFIED
    if status == "escalated":
        return ESCALATED
    if response.get("exit_code") not in (None, 0):
        return FAILED
    return NONE


def _emit_outcome(
    *, call_id: str | None = None, executed_command: str | None = None,
    verification: str = NONE, status: str | None = None, response: dict | None = None,
    recovery_command: str | None = None, graph_edge: dict | None = None,
    decision: str | None = None, executed: list | None = None,
) -> None:
    """Fire-and-forget outcome row; DecisionJournal.record_outcome is fail-open,
    so telemetry can never break the action path it observes."""
    reasons = exit_code = satisfied = None
    if response is not None:
        status = response.get("status", status)
        reasons = response.get("reasons")
        exit_code = response.get("exit_code")
        satisfied = response.get("satisfied")
    goal_id, goal = decision_log.current_goal()
    decision_log.get_journal().record_outcome(
        call_id=call_id, executed_command=executed_command, verification=verification,
        status=status, recovery_command=recovery_command, graph_edge=graph_edge,
        device=transport.serial, decision=decision, reasons=reasons,
        exit_code=exit_code, satisfied=satisfied, goal=goal, goal_id=goal_id,
        executed=executed,
    )


async def _foreground_safe() -> str | None:
    """Foreground package for graph_edge rows; a failed dump is telemetry loss,
    never an execution failure."""
    try:
        return foreground_package(await dump_screen(transport))
    except Exception:  # noqa: BLE001 -- telemetry only
        return None


def _toggle_response(outcome: ToggleOutcome) -> dict:
    return {
        # Weakest-link status: a clean resolve with a low post-verify `satisfied` still isn't "ok".
        "status": "ok" if not outcome.reasons and outcome.satisfied >= current_profile(jev.engine_name).noul_floor else "unverified",
        "exit_code": outcome.exit_code,
        "check_service": outcome.check_service,
        "satisfied": outcome.satisfied,
        "reasons": list(outcome.reasons),
    }


def _tap_response(outcome: ActionOutcome) -> dict:
    return {
        "status": "ok" if outcome.tapped and outcome.satisfied >= current_profile(jev.engine_name).noul_floor else "unverified",
        "element": outcome.element,
        "satisfied": outcome.satisfied,
        "reasons": list(outcome.reasons),
    }


def _launch_response(result: LaunchOutcome) -> dict:
    return {
        "status": "ok" if result.launched else "escalated",
        "package": result.package,
        "satisfied": result.satisfied,
        "reasons": list(result.reasons),
    }


def _dumpsys_response(outcome: DumpsysOutcome) -> dict:
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


def _exit_code_response(exit_code: int) -> dict:
    return {"status": "ok" if exit_code == 0 else "unverified", "exit_code": exit_code}


def _scroll_response(outcome) -> dict:
    return {
        "status": "ok" if outcome.found else "escalated",
        "found": outcome.found,
        "attempts": outcome.attempts,
        "reasons": list(outcome.reasons),
    }


def _resolve_command(proposal, auto_approve: bool) -> CommandVariant | None:
    """auto_approve only decides what happens after a needs_approval verdict --
    a DENIED command (proposal.pending is None and proposal.ready is None) still
    stops the action regardless, same invariant as everywhere else in this codebase."""
    if proposal.ready is not None:
        return proposal.ready
    if auto_approve and proposal.pending is not None:
        return proposal.pending.command
    return None


# The response builder per KIND_TABLE entry -- the MCP-specific half of the shared dispatch
# (dispatch.KIND_TABLE covers propose/execute/resume_arg; CLI needs no dict response at all).
_RESPONSE_FOR = {
    "toggle_service": _toggle_response,
    "tap": _tap_response,
    "long_press": _tap_response,
    "type_text": _tap_response,
    "keyevent": _exit_code_response,
    "swipe": _exit_code_response,
    "set_dnd": _exit_code_response,
}


@dataclass
class _ResumeProposal:
    """Adapts a stored PendingAction back into the shape KIND_TABLE's execute lambdas expect --
    each one reads only its own kind's field (element or service), never both."""
    element: object
    service: object
    confidence: float


async def _do_gated(kind: str, goal: str, *, verify: bool = True, auto_approve: bool = False) -> dict:
    t0 = time.monotonic()
    handler = KIND_TABLE[kind]
    proposal = await handler.propose(jev, transport, goal)
    command = _resolve_command(proposal, auto_approve)
    if command is None:
        if proposal.pending is not None:
            # Proposed but awaiting a human decision: an escalation row now, the
            # execution row later from device_approve (same call_id joins them).
            _emit_outcome(call_id=_call_id_of(proposal), verification=ESCALATED, status="needs_approval")
            return _store_pending(goal, kind, handler.resume_arg(proposal), getattr(proposal, "confidence", 0.0), proposal.pending, verify)
        _emit_outcome(call_id=_call_id_of(proposal), verification=ESCALATED, status="escalated")
        return {"status": "escalated", "reasons": list(proposal.reasons)}
    outcome = await handler.execute(jev, transport, goal, proposal, command, verify=verify, verbose=False)
    response = _RESPONSE_FOR[kind](outcome)
    _emit_outcome(call_id=_call_id_of(proposal), executed_command=command.command,
                  verification=_verification_from_response(response), response=response)
    if kind in ("tap", "long_press", "type_text"):
        response = {**response, "elapsed_s": round(time.monotonic() - t0, 2)}
    return response


async def _do_open_app(goal: str, **_kw) -> dict:
    result = await launch_app_for_goal(jev, transport, goal, verbose=False)
    response = _launch_response(result)
    # Ungated-kind coverage: these kinds emit their own outcome rows here (P1
    # follow-up, landed P7) so trajectories join by call_id like every other kind.
    _emit_outcome(
        call_id=result.call_id,
        executed_command=f"monkey -p {result.package} 1" if result.package else None,
        verification=_verification_from_response(response),
        response=response,
    )
    return response


async def _do_dumpsys(goal: str, **_kw) -> dict:
    result = await run_dumpsys_query(jev, transport, goal, verbose=False)
    response = _dumpsys_response(result)
    _emit_outcome(
        call_id=result.call_id,
        executed_command=f"dumpsys {result.service}" if result.service else None,
        verification=_verification_from_response(response),
        response=response,
    )
    return response


async def _do_scroll_to_find(goal: str, *, direction: str = "down", max_attempts: int = 8, **_kw) -> dict:
    result = await scroll_to_find(jev, transport, goal, direction=direction, max_attempts=max_attempts, verbose=False)
    response = _scroll_response(result)
    _emit_outcome(
        call_id=result.call_id,
        executed_command=result.executed[-1] if result.executed else None,
        verification=_verification_from_response(response),
        response=response,
        executed=list(result.executed),
    )
    return {**response, "executed": list(result.executed)}


async def _do_screenshot(goal: str, **_kw) -> dict:
    response = {"status": "ok"}
    _emit_outcome(call_id=None, executed_command="screencap -p",
                  verification=_verification_from_response(response), response=response)
    return response


# Kinds outside KIND_TABLE's propose->gate->execute shape -- same special-casing run_toolkit uses.
_UNGATED_DISPATCH = {
    "open_app": _do_open_app,
    "dumpsys": _do_dumpsys,
    "scroll_to_find": _do_scroll_to_find,
    "screenshot": _do_screenshot,
}

async def _with_screenshot(response: dict, *, include: bool) -> list:
    """A screenshot embeds a full real image in the response -- real context cost, so it's
    attached only when the caller actually asks for one, not on every action by default."""
    if not include:
        return [response]
    return [response, Image(data=await take_screenshot(transport), format="png")]


@mcp.tool()
async def device_do(
    goal: str, verify: bool = True, auto_approve: bool = False,
    direction: str = "down", max_attempts: int = 8, include_screenshot: bool = False,
) -> list:
    """One tool for any single real phone action. Jev first picks which ONE atomic kind this
    goal wants -- open an app, tap, long-press, type, swipe, scroll-to-find, press a key, toggle
    a service, set DND, take a screenshot, or read a system service (the same ACTION_KINDS the
    CLI toolkit uses) -- then runs exactly that one real action and returns its result.
    include_screenshot=True also attaches a real screenshot of the resulting screen -- off by
    default since an image costs real context; ask for one at a checkpoint, not after every
    single step in a sequence. Never plans or chains more than one action; sequencing multiple
    goals is still the calling agent's job."""
    with goal_scope(goal):
        kind_pick = await pick_kind(jev, goal, transport, verbose=False)
        kind = kind_pick.kind
        if kind is None:
            _emit_outcome(call_id=kind_pick.call_id, verification=ESCALATED, status="escalated")
            return await _with_screenshot({"status": "escalated", "reasons": list(kind_pick.reasons)}, include=include_screenshot)
        # A goal that IS a screenshot returns the image even with include_screenshot=False --
        # otherwise device_do(goal="take a screenshot") would answer {"status": "ok"} and nothing else.
        include = include_screenshot or kind == "screenshot"
        if kind in KIND_TABLE:
            response = await _do_gated(kind, goal, verify=verify, auto_approve=auto_approve)
        else:
            response = await _UNGATED_DISPATCH[kind](goal, verify=verify, auto_approve=auto_approve, direction=direction, max_attempts=max_attempts)
        return await _with_screenshot(response, include=include)


@mcp.tool()
async def device_screenshot() -> Image:
    """Take a screenshot of the current screen and return the actual image --
    never gated (read-only), no goal needed, no candidates to narrow."""
    return Image(data=await take_screenshot(transport), format="png")


@mcp.tool()
async def device_approve(thread_id: str, decision: str, command: str | None = None, include_screenshot: bool = False) -> list:
    """Resolve a pending action from any device_* tool. decision is "approve"
    or "deny". An optional `command` overrides the proposed command (e.g. a
    human-corrected variant), matching PLAN.md's device_approve(thread_id,
    decision, command?) signature. include_screenshot=True attaches a real screenshot,
    same off-by-default tradeoff as device_do."""
    action = _PENDING.pop(thread_id, None)
    if action is None:
        return await _with_screenshot({"status": "error", "reason": f"unknown or already-resolved thread_id {thread_id!r}"}, include=include_screenshot)
    with goal_scope(action.goal):
        if decision != "approve":
            _emit_outcome(call_id=action.call_id, verification=NONE, status="not_executed", decision="deny")
            return await _with_screenshot({"status": "not_executed"}, include=include_screenshot)

        approved = action.pending.command
        recovery_command = None
        if command:
            approved = CommandVariant(command=command, rationale=approved.rationale)
            # A human-corrected command is recorded as a recovery input.
            recovery_command = approved.command

        before = await _foreground_safe()
        handler = KIND_TABLE[action.kind]
        resume_proposal = _ResumeProposal(element=action.resume_arg, service=action.resume_arg, confidence=action.confidence)
        outcome = await handler.execute(jev, transport, action.goal, resume_proposal, approved, verify=action.verify, verbose=False)
        response = _RESPONSE_FOR[action.kind](outcome)
        after = await _foreground_safe()
        graph_edge = {"from_node": before, "to_node": after} if (before or after) else None
        _emit_outcome(
            call_id=action.call_id, executed_command=approved.command,
            verification=_verification_from_response(response), response=response,
            recovery_command=recovery_command, graph_edge=graph_edge, decision="approve",
        )
        return await _with_screenshot(response, include=include_screenshot)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
