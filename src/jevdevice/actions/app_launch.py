"""Launch a real installed app for a goal: real package listing -> two-round semantic narrow
(narrowing.py) -> gate -> `monkey` launch -> Jev-verified retry against the real foreground app.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from jevdevice import question_sets
from jevdevice.budget import current_profile
from jevdevice.device import Device
from jevdevice.jev import JudgeEngine, ask
from jevdevice.judge.narrowing import narrow_and_pick

from .elements import dump_screen, foreground_package


def parse_package_list(raw: str) -> list[str]:
    return [line.removeprefix("package:").strip() for line in raw.splitlines() if line.startswith("package:")]


async def verify_with_retry(
    jev: JudgeEngine, device: Device, goal: str, chosen: str,
    delays: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> tuple[bool, float, int, str]:
    """Real launches race a settling UI; retry with backoff. `dumpsys window`'s
    single mCurrentFocus line can be a nameless system popup mid-launch, so the
    foreground app is read from a real uiautomator dump instead."""
    satisfied = 0.0
    observed = None
    for attempt, delay in enumerate(delays, start=1):
        await asyncio.sleep(delay)
        observed = foreground_package(await dump_screen(device))
        _, answers = await ask(
            jev,
            {"goal": goal, "chosen_package": chosen, "foreground_package": observed, "attempt": attempt},
            {"satisfied": question_sets.noul("open_app.verify_satisfied")},
            phase="verify",
        )
        satisfied = answers["satisfied"].noul
        if satisfied >= current_profile(jev.name).noul_floor:
            return True, satisfied, attempt, observed or ""
    return False, satisfied, len(delays), observed or ""


@dataclass
class LaunchOutcome:
    package: str | None
    confidence: float
    launched: bool
    satisfied: float
    attempts: int
    observed: str
    reasons: tuple[str, ...] = ()
    # Journal linkage: the pick's round-2 ground ask, so the outcome row (emitted
    # by mcp_server for this ungated kind) joins the decision row that chose it.
    call_id: str | None = None
    label_keys: tuple[str, ...] = ()  # the executed launch pick's fit_i
    gate_key: str | None = None  # ungated: always None


async def launch_app_for_goal(jev: JudgeEngine, device: Device, goal: str, *, verbose: bool = True) -> LaunchOutcome:
    """Real package listing -> two-round semantic narrow (narrowing.py) -> gate
    -> `monkey` launch -> Jev-verified retry."""
    probe = await device.run("pm list packages")
    packages = parse_package_list(probe.stdout)

    verdict = await narrow_and_pick(
        jev, goal, packages,
        instructions=question_sets.text("open_app.pick"),
        fit_instructions=question_sets.text("open_app.fit"),
        pick_qid="open_app.pick", fit_qid="open_app.fit",
    )
    if verbose:
        print(f"shortlist: {verdict.shortlist}")
        print(f"Jev picked: {verdict.choice} (confidence {verdict.confidence:.2f}, fit {verdict.fit:.2f})")
    label_keys = (verdict.fit_key,) if verdict.fit_key else ()
    if not verdict.ok:
        if verbose:
            print(f"=== ESCALATED === {'; '.join(verdict.reasons)}")
        return LaunchOutcome(None, verdict.confidence, False, 0.0, 0, "", tuple(verdict.reasons), call_id=verdict.call_id, label_keys=label_keys)
    package = verdict.choice

    await device.run(f"monkey -p {package} 1")
    launched, satisfied, attempts, observed = await verify_with_retry(jev, device, goal, package)
    if verbose:
        print(f"foreground: {observed}")
        print(f"goal met: {launched} (noul={satisfied:.2f}, after {attempts} attempt(s))")
    return LaunchOutcome(package, verdict.confidence, launched, satisfied, attempts, observed, call_id=verdict.call_id, label_keys=label_keys)
