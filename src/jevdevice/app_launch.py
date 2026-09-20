"""Launch a real installed app for a goal: real package listing -> two-round semantic narrow
(narrowing.py) -> gate -> `monkey` launch -> Jev-verified retry against the real foreground app.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .budget import current_profile
from .elements import dump_screen, foreground_package
from .jev import JevClient, Noul
from .narrowing import narrow_and_pick
from .transport import AdbTransport


def parse_package_list(raw: str) -> list[str]:
    return [line.removeprefix("package:").strip() for line in raw.splitlines() if line.startswith("package:")]


async def verify_with_retry(
    jev: JevClient, transport: AdbTransport, goal: str, chosen: str,
    delays: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> tuple[bool, float, int, str]:
    """Real launches race a settling UI; retry with backoff. `dumpsys window`'s
    single mCurrentFocus line can be a nameless system popup mid-launch, so the
    foreground app is read from a real uiautomator dump instead."""
    satisfied = 0.0
    observed = None
    for attempt, delay in enumerate(delays, start=1):
        await asyncio.sleep(delay)
        observed = foreground_package(await dump_screen(transport))
        answers = await jev.ask(
            {"goal": goal, "chosen_package": chosen, "foreground_package": observed, "attempt": attempt},
            {"satisfied": Noul(instructions="Is foreground_package the app named by chosen_package, or otherwise "
                                             "evidence that the goal is now achieved for chosen_package?")},
            phase="verify",
        )
        satisfied = answers["satisfied"].noul
        if satisfied >= current_profile(jev.engine_name).noul_floor:
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


async def launch_app_for_goal(jev: JevClient, transport: AdbTransport, goal: str, *, verbose: bool = True) -> LaunchOutcome:
    """Real package listing -> two-round semantic narrow (narrowing.py) -> gate
    -> `monkey` launch -> Jev-verified retry."""
    probe = await transport.run("pm list packages")
    packages = parse_package_list(probe.stdout)

    verdict = await narrow_and_pick(
        jev, goal, packages,
        instructions="Which package best satisfies the goal?",
        fit_instructions="Is {candidate} the app the goal asks to open?",
    )
    if verbose:
        print(f"shortlist: {verdict.shortlist}")
        print(f"Jev picked: {verdict.choice} (confidence {verdict.confidence:.2f}, fit {verdict.fit:.2f})")
    if not verdict.ok:
        if verbose:
            print(f"=== ESCALATED === {'; '.join(verdict.reasons)}")
        return LaunchOutcome(None, verdict.confidence, False, 0.0, 0, "", tuple(verdict.reasons))
    package = verdict.choice

    await transport.run(f"monkey -p {package} 1")
    launched, satisfied, attempts, observed = await verify_with_retry(jev, transport, goal, package)
    if verbose:
        print(f"foreground: {observed}")
        print(f"goal met: {launched} (noul={satisfied:.2f}, after {attempts} attempt(s))")
    return LaunchOutcome(package, verdict.confidence, launched, satisfied, attempts, observed)
