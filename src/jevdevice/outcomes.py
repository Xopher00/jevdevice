"""Shared outcome-row emission for every flow that runs a real action:
mcp_server.py (device_do/device_approve), the P7.5 planner, and any future
caller. One path instead of independently-maintained switches that drift.

`emit_outcome` is fail-open by construction -- DecisionJournal.record_outcome
catches and prints -- so telemetry can never break the action it observes.
`graph_edge` (foreground package before/after, guarded) is the P7.5 recipe
primitive: a recipe step is an ordered graph_edge sequence. It costs two
resident-uiautomator2 dumps (~0.6 s), so it is knob-gated (JEV_GRAPH_EDGE,
default on) and always telemetry-loss-only: a failed dump is None, never an
execution failure.
"""

from __future__ import annotations

import os

from . import decision_log
from .elements import dump_screen, foreground_package

ENV_GRAPH_EDGE = "JEV_GRAPH_EDGE"  # "0"/"off" disables before/after foreground dumps
GRAPH_EDGE_DEFAULT = True  # recipes (P7.5) need edges on real runs; knob exists for latency-critical use


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
    """Foreground package for graph_edge rows; a failed dump is telemetry loss,
    never an execution failure."""
    try:
        return foreground_package(await dump_screen(transport))
    except Exception:  # noqa: BLE001 -- telemetry only
        return None


async def graph_edge_around(transport, run) -> tuple:
    """Run `run()`, bracketing it with foreground dumps -> (outcome, edge).
    edge is None when either side is missing or the knob is off -- telemetry
    loss, never an execution failure."""
    if not graph_edge_enabled():
        return await run(), None
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
    """Fire-and-forget outcome row with the full shared field set (the additive
    P7.5 fields kind/tier/recipe_id included)."""
    reasons = exit_code = satisfied = None
    if response is not None:
        status = response.get("status", status)
        exit_code = response.get("exit_code")
        satisfied = response.get("satisfied")
    reasons = reasons if reasons is not None else response.get("reasons") if response is not None else None
    goal_id, goal = decision_log.current_goal()
    decision_log.get_journal().record_outcome(
        call_id=call_id, executed_command=executed_command, verification=verification,
        status=status, recovery_command=recovery_command, graph_edge=graph_edge,
        device=getattr(transport, "serial", None) if transport is not None else None, decision=decision,
        reasons=reasons, exit_code=exit_code, satisfied=satisfied,
        goal=goal, goal_id=goal_id, executed=executed,
        kind=kind, tier=tier, recipe_id=recipe_id,
    )
