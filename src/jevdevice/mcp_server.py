"""MCP server: device_do picks and runs exactly ONE atomic action kind per call (Jev's own
Choice over ACTION_KINDS, the same one the CLI toolkit uses) -- device_screenshot and
device_approve are separate since they don't fit that shape (a real image; resuming a pending
action). The calling agent (an LLM composing these calls) still does all multi-step planning
and sequencing; device_do never chains more than one action, it only cuts the tool count the
agent has to pick from for that one action.

Execution itself lives in dispatch.run_kind (shared with the planner): propose -> gate ->
execute, one journaled outcome row per action. This module only adds the MCP-specific
half -- picking the kind per call, attaching screenshots on request, and the
needs_approval -> device_approve resume flow.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image

from jevdevice.actions.services import take_screenshot
from jevdevice.execution.dispatch import (
    ATTACHES_RESULT,
    KIND_TABLE,
    pick_kind,
    response_for,
    run_kind,
)
from jevdevice.journal import outcomes
from jevdevice.journal.decision_log import goal_scope
from jevdevice.journal.outcomes import LabelTarget
from jevdevice.judge.gate import CommandVariant, Pending

from .common import bootstrap

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
    label: LabelTarget | None = None
    tier: int | None = None
    recipe_id: str | None = None


_PENDING: dict[str, PendingAction] = {}


def _pending_response(action_id: str, action: PendingAction) -> dict:
    return {
        "status": "needs_approval",
        "thread_id": action_id,
        "command": action.pending.command.command,
        "rationale": action.pending.command.rationale,
        "confidence": action.pending.gate_result.confidence,
    }


def _store_pending(
    goal: str, kind: str, resume_arg: str | None, confidence: float, pending: Pending,
    verify: bool = True, *, label: LabelTarget | None = None, tier: int | None = None,
    recipe_id: str | None = None,
) -> dict:
    """dispatch.run_kind's on_pending hook: park the proposed action for a
    human and surface the approval prompt as the tool response."""
    action_id = str(uuid.uuid4())
    action = PendingAction(
        goal, kind, resume_arg, confidence, pending, verify,
        call_id=pending.gate_result.call_id if pending.gate_result else None,
        label=label, tier=tier, recipe_id=recipe_id,
    )
    _PENDING[action_id] = action
    return _pending_response(action_id, action)


def _emit_outcome(**kw) -> None:
    """Act/Verify journaling bound to this server's device (outcomes.record_action
    does the work; journaling is fail-open). Stays named `transport`: a test
    reads mcp_server.transport.serial."""
    outcomes.record_action(device=transport, **kw)


@dataclass
class _ResumeProposal:
    """Adapts a stored PendingAction back into the shape KIND_TABLE's execute lambdas expect --
    each one reads only its own kind's field (element or service), never both."""
    element: object
    service: object
    confidence: float


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
            _emit_outcome(call_id=kind_pick.call_id, response={"status": "escalated"})
            return await _with_screenshot({"status": "escalated", "reasons": list(kind_pick.reasons)}, include=include_screenshot)
        include = include_screenshot or kind in ATTACHES_RESULT
        response = await run_kind(
            jev, transport, kind, goal, verify=verify, auto_approve=auto_approve,
            direction=direction, max_attempts=max_attempts, on_pending=_store_pending,
        )
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
    human-corrected variant). include_screenshot=True attaches a real screenshot,
    same off-by-default tradeoff as device_do."""
    action = _PENDING.pop(thread_id, None)
    if action is None:
        return await _with_screenshot({"status": "error", "reason": f"unknown or already-resolved thread_id {thread_id!r}"}, include=include_screenshot)
    with goal_scope(action.goal):
        if decision != "approve":
            _emit_outcome(call_id=action.call_id, key=action.kind, kind=action.kind, gate=action.pending.gate_result, status="not_executed", decision="deny")
            return await _with_screenshot({"status": "not_executed"}, include=include_screenshot)

        approved = action.pending.command
        recovery_command = None
        if command:
            approved = CommandVariant(command=command, rationale=approved.rationale)
            # A human-corrected command is recorded as a recovery input.
            recovery_command = approved.command

        before = await outcomes.foreground_safe(transport)
        handler = KIND_TABLE[action.kind]
        resume_proposal = _ResumeProposal(element=action.resume_arg, service=action.resume_arg, confidence=action.confidence)
        outcome = await handler.execute(jev, transport, action.goal, resume_proposal, approved, verify=action.verify, verbose=False)
        response = response_for(action.kind, outcome, jev.name)
        after = await outcomes.foreground_safe(transport)
        graph_edge = {"from_node": before, "to_node": after} if (before or after) else None
        if action.label is not None:
            _emit_outcome(
                label=action.label, kind=action.kind, gate=action.pending.gate_result,
                executed_command=approved.command, response=response,
                recovery_command=recovery_command, graph_edge=graph_edge, decision="approve",
                tier=action.tier, recipe_id=action.recipe_id,
            )
        else:
            _emit_outcome(
                call_id=action.call_id, key=action.kind, kind=action.kind, gate=action.pending.gate_result,
                executed_command=approved.command, response=response,
                recovery_command=recovery_command, graph_edge=graph_edge, decision="approve",
                tier=action.tier, recipe_id=action.recipe_id,
            )
        return await _with_screenshot(response, include=include_screenshot)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
