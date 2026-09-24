"""Shared Act/Verify journaling for every flow that runs a real action (the MCP
server, the planner): one path through core's `record_outcome`/`record_verdict`.

A `graph_edge` records the foreground app before and after an action -- the
unit stored-recipe chains are built from. Knob-gated (JEV_GRAPH_EDGE, default
on) and telemetry-only: a failed dump yields None, never fails the action.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar

from typesymbolic.domain import ActOutcome, ActStep, Verdict

from jevdevice.actions.elements import dump_screen, foreground_package
from jevdevice.device import Device

from . import decision_log

ENV_GRAPH_EDGE = "JEV_GRAPH_EDGE"  # "0"/"off" disables the before/after foreground dumps
GRAPH_EDGE_DEFAULT = True

# One run's episode_id -- record_action reads it like goal_scope()'s goal, no threading needed.
_EPISODE: ContextVar[str | None] = ContextVar("jevdevice_episode", default=None)


@contextmanager
def episode_scope(episode_id: str | None):
    token = _EPISODE.set(episode_id)
    try:
        yield
    finally:
        _EPISODE.reset(token)


def graph_edge_enabled() -> bool:
    return os.environ.get(ENV_GRAPH_EDGE, "1" if GRAPH_EDGE_DEFAULT else "0").strip().lower() not in {"0", "off", "false", "no"}


async def foreground_safe(device: Device) -> str | None:
    """Foreground app name for graph_edge rows; None when the knob is off or
    the dump fails (telemetry loss, never an execution failure)."""
    if not graph_edge_enabled():
        return None
    try:
        return foreground_package(await dump_screen(device))
    except Exception:  # noqa: BLE001 -- telemetry only
        return None


async def graph_edge_around(device: Device, run) -> tuple:
    """Run `run()` bracketed with foreground dumps -> (outcome, edge). The
    edge is None when the knob is off or either dump fails."""
    before = await foreground_safe(device)
    outcome = await run()
    after = await foreground_safe(device)
    edge = {"from_node": before, "to_node": after} if (before or after) else None
    return outcome, edge


def verdict_from_response(response: dict, *, key: str | None = None) -> Verdict:
    """The flow's own status -> a device Verdict; `key` is the pick key, never the `safe` gate noul."""
    status = response.get("status")
    if status == "ok":
        result = "verified"
    elif status == "escalated":
        result = "escalated"
    elif response.get("exit_code") not in (None, 0):
        result = "failed"
    else:
        result = "unconfirmed"
    return Verdict(status=result, tests=(key,) if key else ())


def record_action(
    *, device: Device | None = None, call_id: str | None = None, key: str | None = None,
    gate=None, response: dict | None = None, executed_command: str | None = None,
    executed: list | None = None, recovery_command: str | None = None,
    graph_edge: dict | None = None, decision: str | None = None, status: str | None = None,
    tier: int | None = None, recipe_id: str | None = None, reasons=None, succeeded: bool | None = None,
) -> Verdict | None:
    """Journal one Act (`record_outcome`, `key` = the kind/element pick key) and,
    given a device `response`, the Verdict it implies (`record_verdict`), both
    on `call_id`. `executed` -> one ActStep per command. Returns the Verdict."""
    satisfied = exit_code = None
    if response is not None:
        status = response.get("status", status)
        satisfied, exit_code = response.get("satisfied"), response.get("exit_code")
        reasons = reasons if reasons is not None else response.get("reasons")
    verdict = verdict_from_response(response, key=key) if response is not None else None
    succeeded = succeeded if succeeded is not None else bool(verdict and verdict.status == "verified")
    goal_id, goal = decision_log.current_goal()
    outcome = ActOutcome(
        succeeded=succeeded, key=key, reasons=tuple(reasons or ()),
        steps=tuple(ActStep(name=command, succeeded=True) for command in (executed or ())),
    )
    extra = {k: v for k, v in {
        "device": getattr(device, "name", None) if device is not None else None,
        "goal": goal, "goal_id": goal_id, "executed_command": executed_command,
        "recovery_command": recovery_command, "graph_edge": graph_edge, "decision": decision,
        "status": status, "tier": tier, "recipe_id": recipe_id,
        "satisfied": satisfied, "exit_code": exit_code,
    }.items() if v is not None}
    journal = decision_log.get_journal()
    journal.record_outcome(call_id=call_id, gate=gate, outcome=outcome, extra=extra or None,
                           episode_id=_EPISODE.get())
    if verdict:
        journal.record_verdict(call_id=call_id, verdict=verdict)
    return verdict
