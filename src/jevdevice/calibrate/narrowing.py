"""Diagnostic: runs narrowing.narrow_and_pick() against hand-picked obvious-fit/
near-tie/genuinely-absent goals over the real device's package and dumpsys-service
enumerations, to check whether min_fit and the chunk-scoring formula actually
separate the three categories, or whether the scale sits near the 0.5 threshold.
"""

from __future__ import annotations

import asyncio

from jevdevice.app_launch import parse_package_list
from jevdevice.common import bootstrap
from jevdevice.narrowing import narrow_and_pick
from jevdevice.services import parse_dumpsys_services

PACKAGE_CASES = [
    ("obvious-fit", "open Gmail"),
    ("obvious-fit, zero lexical overlap", "check my email"),
    ("near-tie", "open messages"),
    ("near-tie", "take a photo"),
    ("genuinely-absent", "book me a flight to Paris"),
    ("genuinely-absent", "order a pizza"),
]

SERVICE_CASES = [
    ("obvious-fit", "what is my battery level?"),
    ("obvious-fit", "am I connected to wifi?"),
    ("near-tie", "what's using the most memory?"),
    ("genuinely-absent", "what's the weather today?"),
    ("genuinely-absent", "turn off airplane mode"),
]


async def run_cases(jev, candidates, cases, *, instructions, fit_instructions):
    for category, goal in cases:
        verdict = await narrow_and_pick(jev, goal, candidates, instructions=instructions, fit_instructions=fit_instructions)
        print(f"{category:<32} {goal!r:<32} ok={verdict.ok!s:<6} choice={verdict.choice} "
              f"confidence={verdict.confidence:.2f} fit={verdict.fit:.2f} best_fit={verdict.best_fit:.2f}")
        if not verdict.ok:
            print(f"{'':<32} {'':<32} reasons: {verdict.reasons}")


async def main() -> None:
    jev, transport = bootstrap()

    packages_raw = await transport.run("pm list packages")
    packages = parse_package_list(packages_raw.stdout)
    print(f"=== packages ({len(packages)} real) ===")
    await run_cases(
        jev, packages, PACKAGE_CASES,
        instructions="Which package best satisfies the goal?",
        fit_instructions="Is {candidate} the app the goal asks to open?",
    )

    services_raw = await transport.run("dumpsys -l")
    services = parse_dumpsys_services(services_raw.stdout)
    print(f"\n=== dumpsys services ({len(services)} real) ===")
    await run_cases(
        jev, services, SERVICE_CASES,
        instructions="Which service would answer this goal?",
        fit_instructions="Would `dumpsys {candidate}` actually contain the answer to the goal?",
    )


if __name__ == "__main__":
    asyncio.run(main())
