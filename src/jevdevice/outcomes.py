"""Shared outcome-row emission for every flow that runs a real action (the
MCP server, the planner): one journaling path instead of per-caller copies
that drift.

A `graph_edge` records the foreground app before and after an action -- the
unit stored-recipe chains are built from. It costs two screen dumps, so it
is knob-gated (JEV_GRAPH_EDGE, default on) and telemetry-only: a failed
dump yields None and never fails the action it observes.
"""

from __future__ import annotations

import os

from . import decision_log
from .elements import dump_screen, foreground_package

ENV_GRAPH_EDGE = "JEV_GRAPH_EDGE"  # "0"/"off" disables the before/after foreground dumps
GRAPH_EDGE_DEFAULT = True


def graph_edge_enabled() -> bool:
    return os.environ.get(ENV_GRAPH_EDGE, "1" if GRAPH_EDGE_DEFAULT else "0").strip().lower() not in {"0", "off", "false", "no"}


def verification_from_response(response: dict) -> str:
    """The flow's own status -> journal verification. ok = verified; a non-zero
    exit is an outright failure; anything else that ran is unconfirmed (none)."""
    status = response.get("status")
    if status == "ok":
        return decision_log.VERIFIED
    if status == "escalated":
        return decision_log.ESCALATED
    if response.get("exit_code") not in (None, 0):
        return decision_log.FAILED
    return decision_log.NONE


async def foreground_safe(transport) -> str | None:
    """Foreground app name for graph_edge rows; None when the knob is off or
    the dump fails (telemetry loss, never an execution failure)."""
    if not graph_edge_enabled():
        return None
    try:
        return foreground_package(await dump_screen(transport))
    except Exception:  # noqa: BLE001 -- telemetry only
        return None


async def graph_edge_around(transport, run) -> tuple:
    """Run `run()` bracketed with foreground dumps -> (outcome, edge). The
    edge is None when the knob is off or either dump fails."""
    before = await foreground_safe(transport)
    outcome = await run()
    after = await foreground_safe(transport)
    edge = {"from_node": before, "to_node": after} if (before or after) else None
    return outcome, edge


def emit_outcome(
    *, transport=None, call_id: str | None = None, executed_command: str | None = None,
    verification: str = decision_log.NONE, status: str | None = None, response: dict | None = None,
    recovery_command: str | None = None, graph_edge: dict | None = None,
    decision: str | None = None, executed: list | None = None,
    kind: str | None = None, tier: int | None = None, recipe_id: str | None = None,
    reasons=None,
) -> None:
    """Fire-and-forget outcome row. `response`, when given, supplies the
    status/reasons/exit_code/satisfied fields verbatim (explicit `reasons`
    wins); `kind` names the action kind that ran; `tier`/`recipe_id` appear
    only on planner-driven rows."""
    exit_code = satisfied = None
    if response is not None:
        status = response.get("status", status)
        exit_code = response.get("exit_code")
        satisfied = response.get("satisfied")
        if reasons is None:
            reasons = response.get("reasons")
    goal_id, goal = decision_log.current_goal()
    decision_log.get_journal().record_outcome(
        call_id=call_id, executed_command=executed_command, verification=verification,
        status=status, recovery_command=recovery_command, graph_edge=graph_edge,
        device=getattr(transport, "serial", None), decision=decision,
        reasons=reasons, exit_code=exit_code, satisfied=satisfied,
        goal=goal, goal_id=goal_id, executed=executed,
        kind=kind, tier=tier, recipe_id=recipe_id,
    )
