"""Step 0: prove the core mechanism on one real case, real phone, real Jev.

No mocks, no smallmodel.py, no grounding.py, no gate.py, no recipes.py, no loop.py.
Candidates come from deterministic code parsing + a generic fuzzy-match narrow,
never a model. Jev makes both decisions: which candidate, and whether the goal
was met afterward.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jevdevice.jev import Choice, JevClient, Noul
from jevdevice.transport import AdbTransport

SERIAL = "192.168.100.11:46585"
GOAL = sys.argv[1] if len(sys.argv) > 1 else "open the Calculator app"


def parse_package_list(raw: str) -> list[str]:
    return [line.removeprefix("package:").strip() for line in raw.splitlines() if line.startswith("package:")]


def fuzzy_narrow(goal: str, packages: list[str], limit: int = 20) -> list[str]:
    """Score each package by its best-matching dot-separated segment, not the
    whole reverse-DNS string, so "calculator" can match ...app.popupcalculator."""
    goal_tokens = [t for t in goal.casefold().split() if len(t) > 3]

    def score(pkg: str) -> float:
        segments = pkg.casefold().replace("_", ".").split(".")
        return max(
            (difflib.SequenceMatcher(None, token, segment).ratio() for token in goal_tokens for segment in segments),
            default=0.0,
        )

    scored = sorted(((score(pkg), pkg) for pkg in packages), reverse=True)
    return [pkg for _, pkg in scored[:limit]]


def confidence_gate(probabilities: dict[str, float], confidence: float, min_confidence: float = 0.6, min_margin: float = 0.15) -> tuple[bool, str]:
    """No action below threshold — escalate instead of acting on a close or unsure pick."""
    ranked = sorted(probabilities.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) >= 2 else ranked[0]
    if confidence < min_confidence:
        return False, f"confidence {confidence:.2f} below {min_confidence}"
    if margin < min_margin:
        return False, f"margin {margin:.2f} below {min_margin} (top two picks too close)"
    return True, "ok"


async def verify_with_retry(
    jev: JevClient, transport: AdbTransport, goal: str, chosen: str,
    check_command: str = "dumpsys window", delays: tuple[float, ...] = (1.0, 1.5, 2.5),
) -> tuple[bool, float, int, str]:
    """Real launches race a settling UI (e.g. a first-run dialog); retry the
    check with backoff instead of trusting one fixed-delay snapshot."""
    satisfied = 0.0
    focus_line = ""
    for attempt, delay in enumerate(delays, start=1):
        await asyncio.sleep(delay)
        check = await transport.run(check_command)
        focus_line = next((line for line in check.stdout.splitlines() if "mCurrentFocus" in line), "")
        answers = await jev.ask(
            {"goal": goal, "chosen_package": chosen, "current_focus": focus_line, "attempt": attempt},
            {"satisfied": Noul(instructions="Given current_focus, is the goal now achieved for chosen_package? "
                                             "A transient system dialog (e.g. ImmersiveModeConfirmation) during "
                                             "launch is not itself failure or success — judge only the named app's state.")},
        )
        satisfied = answers["satisfied"].noul
        if satisfied >= 0.5:
            return True, satisfied, attempt, focus_line
    return False, satisfied, len(delays), focus_line


async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    transport = AdbTransport(SERIAL)
    jev = JevClient(api_key)

    device = await transport.describe()
    print(f"device: {device}\n")

    print("--- step 1: real probe ---")
    probe = await transport.run("pm list packages")
    print(f"raw output: {len(probe.stdout)} chars, {probe.stdout.count(chr(10))} lines\n")

    print("--- step 2: code parses (no model) ---")
    packages = parse_package_list(probe.stdout)
    print(f"parsed {len(packages)} real packages\n")

    print("--- step 3: generic fuzzy narrow (no model, no per-app table) ---")
    narrowed = fuzzy_narrow(GOAL, packages, limit=20)
    print(f"narrowed to {len(narrowed)}:")
    for p in narrowed:
        print(f"  {p}")
    print()
    assert all(p in probe.stdout for p in narrowed), "narrowed candidate not actually in raw output"

    print("--- step 4: Jev picks (real call) ---")
    criteria = {p: None for p in narrowed}
    answers = await jev.ask(
        {"goal": GOAL, "candidate_packages": narrowed},
        {"pick": Choice(instructions="Which package is the device's Calculator app?", criteria=criteria)},
    )
    pick = answers["pick"]
    print(f"Jev picked: {pick.choice}  (confidence {pick.confidence:.2f})")
    print(f"top probabilities: {sorted(pick.probabilities.items(), key=lambda kv: -kv[1])[:3]}\n")
    assert pick.choice in probe.stdout, "Jev's pick is not a real package (should be structurally impossible)"

    ok, reason = confidence_gate(pick.probabilities, pick.confidence)
    if not ok:
        print(f"=== ESCALATED === {reason}; not acting on an unsure pick")
        return

    print("--- step 5: real action ---")
    act = await transport.run(f"monkey -p {pick.choice} 1")
    print(f"launch exit_code={act.exit_code}\n")

    print("--- step 6: real check probe + Jev verify, with retry ---")
    ok, satisfied, attempts, focus_line = await verify_with_retry(jev, transport, GOAL, pick.choice)
    print(f"focus line: {focus_line.strip()}")
    print(f"Jev verify noul: {satisfied:.2f} (after {attempts} attempt(s))")

    print("\n=== RESULT ===")
    print(f"jev picked: {pick.choice} (confidence {pick.confidence:.2f})")
    print(f"jev verified goal met: {ok}")


if __name__ == "__main__":
    asyncio.run(main())
