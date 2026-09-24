"""Shadow mode: while the primary engine answers live traffic, a second
engine observes the exact same (state, questions) and journals its answers --
and nothing else. Shadow answers are NEVER returned to a caller, NEVER
executed, and NEVER read by the gate: the only thing that escapes it is the
decision row itself (engine="laya", shadow_of=<primary call_id>), written by
the same `jev.ask()` path every other call uses.

Failure isolation is structural, not by convention: the shadow runs in its own
asyncio task whose body is a catch-all -- a shadow crash prints and dies with
the task, so the primary ask() (a different task) cannot observe it. The
shadow's own ask() also journals its failure path (ask_batch emits an error
row in a finally-equivalent except), so even broken shadow calls leave a
paired, labelled row behind.

Knobs (named, env-selected, read at use time so tests can flip them):
- JEV_SHADOW        "1" (default) = attach a laya shadow when the primary
                    engine is jev; "0" disables. The shadow checkpoint loads
                    lazily on the first shadowed ask.
- JEV_SHADOW_MODE   "after" (default) = spawn the shadow ask only after the
                    primary's answer and row are final (zero contention with
                    the in-flight primary call); "concurrent" = spawn it
                    alongside the primary request.
- SHADOW_DRAIN_TIMEOUT_S  aclose() grace for in-flight shadow asks before the
                    remainder are cancelled.

Lifecycle: common.bootstrap() calls attach() on the jev branch, hanging a
`.shadow` JudgeEngine and a `.shadow_tasks` set off the primary judge (plain
attribute assignment -- JevEngine carries no __slots__). jev.ask() generates
its call_id up front and calls schedule() at two points (before_request /
after_answer); aclose() drains. Everything else -- every ask() call site --
inherits shadowing unchanged, because they all go through jev.ask().
"""

from __future__ import annotations

import asyncio
import os

from jevdevice.budget import JEV_ENGINE_NAME

SHADOW_ENV = "JEV_SHADOW"
SHADOW_MODE_ENV = "JEV_SHADOW_MODE"
SHADOW_MODES = ("after", "concurrent")
DEFAULT_SHADOW_MODE = "after"
SHADOW_DRAIN_TIMEOUT_S = 5.0


def enabled() -> bool:
    """Shadow switch: default ON (that is the point of the phase); JEV_SHADOW=0
    (or false/no) turns it off. Values are the operator's, so unknown truthy
    spellings stay off-side safe: only exact on-words enable."""
    return os.environ.get(SHADOW_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def mode() -> str:
    """When the shadow ask fires relative to the primary: 'after' (default) or
    'concurrent'. An unknown value fails fast -- a typo silently picking the
    other mode would corrupt latency comparisons."""
    value = (os.environ.get(SHADOW_MODE_ENV) or DEFAULT_SHADOW_MODE).strip().lower()
    if value not in SHADOW_MODES:
        raise SystemExit(f"{SHADOW_MODE_ENV} must be one of {', '.join(SHADOW_MODES)}, got {value!r}")
    return value


def attach(primary) -> None:
    """Give the primary judge its shadow observer. Jev traffic only: the phase
    shadows laya against live jev decisions (the reverse would need TYPESAFE
    keys for every laya session and teaches nothing new). Idempotent: an
    already-attached shadow is left alone."""
    if getattr(primary, "shadow", None) is not None or primary.name != JEV_ENGINE_NAME or not enabled():
        return
    from jevdevice.common import (  # lazy: common imports this module
        DEFAULT_DEVICE,
        DEVICE_ENV,
    )
    from jevdevice.laya_backend import (
        LayaClient,  # lazy: torch stays inside _build_router
    )

    primary.shadow = LayaClient(device=os.environ.get(DEVICE_ENV, DEFAULT_DEVICE))
    primary.shadow_tasks = set()


def schedule(primary, *, point: str, primary_call_id: str, state: object,
             questions: dict, phase: str | None, truncation: dict | None) -> None:
    """Spawn this ask()'s shadow observation, if the mode wants this point.
    point='before_request' acts only in concurrent mode (shadow starts
    alongside the primary request); point='after_answer' acts only in after
    mode (shadow starts once the primary answer and row are final)."""
    if getattr(primary, "shadow", None) is None:
        return
    if (point == "before_request") != (mode() == "concurrent"):
        return
    task = asyncio.get_running_loop().create_task(_observe(
        primary.shadow, primary_call_id=primary_call_id, state=state,
        questions=questions, phase=phase, truncation=truncation,
    ))
    primary.shadow_tasks.add(task)
    task.add_done_callback(primary.shadow_tasks.discard)


async def _observe(shadow_judge, *, primary_call_id: str, state: object,
                   questions: dict, phase: str | None, truncation: dict | None) -> None:
    """Run one shadow ask, linked back to the primary row via shadow_of. The
    answers stay inside this coroutine -- returning them is the one thing
    this function must never do."""
    from jevdevice.jev import ask

    try:
        await ask(shadow_judge, state, questions, phase=phase,
                 truncation=truncation, shadow_of=primary_call_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 -- a shadow failure must never break the primary path
        print(f"shadow ask failed (primary unaffected): {type(exc).__name__}: {exc}")


async def drain(primary, timeout: float = SHADOW_DRAIN_TIMEOUT_S) -> None:
    """Wait briefly for in-flight shadow asks (aclose()), then cancel the
    remainder: a half-second predict is worth keeping, a 33 s cold checkpoint
    load is not worth a hung shutdown."""
    pending = [task for task in getattr(primary, "shadow_tasks", ()) if not task.done()]
    if not pending:
        return
    _, still_running = await asyncio.wait(pending, timeout=timeout)
    for task in still_running:
        task.cancel()
