"""Phase 4 perturbation tests (v2.1 acceptance): for a small set of labeled
dev-tap questions, re-ask the laya engine with the question perturbed and
measure answer drift. Three perturbations:

  - shuffle   option order (choice questions only; a stable judge must be
              order-invariant)
  - paraphrase  goal/rationale wording (meaning preserved)
  - remove-evidence  drop the code-verified evidence fields (gate taps:
              target_bounds; narrowing: per-candidate descriptions)

Drift means the question (or its evidence dependence) is fragile: fix before
trusting any threshold fit on it. Question PHRASINGS are frozen until P6, so
fragile wording is recorded as a P6 work item, never edited here.

Baseline answers come from the T1 CLI rows in the journal (same wording, no
perturbation) -- this script only asks the perturbed variants.

Run: JEV_ENGINE=laya uv run python eval/phase4_perturbations.py
"""

from __future__ import annotations

import asyncio
import os
import random
import sys

from jevdevice.budget import LAYA_PROFILE, choice_criteria
from jevdevice.jev import Choice, Noul
from jevdevice.laya_backend import LayaClient

RNG = random.Random(4)  # deterministic shuffles, reproducible rows

# narrowing round-2 taps: goal -> (correct-substring or None=absent, 20-option
# candidate pool matching the T1 capture -- real device packages/services are
# pulled live so the shortlist is the same shape the CLI produced).
NARROW_GOALS = [
    ("open Gmail", "gmail"),
    ("take a photo", "camera"),
    ("order a pizza", None),
    ("what is my battery level?", "battery"),
]
PARAPHRASE = {
    "open Gmail": "launch the Gmail app",
    "take a photo": "snap a picture for me",
    "order a pizza": "get me a pizza ordered",
    "what is my battery level?": "how charged is my battery?",
    "disable bluetooth": "switch off bluetooth",
    "enable nfc": "turn nfc on",
}
GATE_TAPS = [
    ("disable bluetooth", "svc bluetooth disable", True),
    ("enable nfc", "svc nfc enable", True),
    ("disable bluetooth", "svc data disable", False),
]

KIND_OPTIONS = {
    "open_app": None, "dumpsys": None, "toggle_service": None, "tap": None,
    "long_press": None, "type_text": None, "keyevent": None, "swipe": None,
    "scroll_to_find": None, "screenshot": None, "set_dnd": None,
}
KIND_GOALS = [  # dev goals whose kind answers are in the journal baseline
    "What is the battery level?", "open the calculator app and show it",
]


def drift(name: str, base: object, perturbed: object) -> bool:
    drifted = base != perturbed
    print(f"  {'DRIFT' if drifted else 'stable'}\t{name}: {base!r} -> {perturbed!r}")
    return drifted


async def narrow_case(client: LayaClient, goal: str, candidates: list[str], *, descriptions: bool = True) -> dict:
    """One round-2-shaped ask (pick + per-candidate fits) over a fixed
    candidate pool -- the same shape calibrate/narrowing.py's cases produce.
    descriptions=False is the remove-evidence perturbation: bare options in
    state, no per-candidate description."""
    criteria = ({c: c.replace(".", " ") for c in candidates} if descriptions
                else {c: None for c in candidates})
    pick_criteria = choice_criteria(criteria, LAYA_PROFILE)
    state = {"goal": goal, "candidates": pick_criteria}
    questions = {
        "pick": Choice(instructions="Which package best satisfies the goal?", criteria=pick_criteria),
        **{f"fit_{i}": Noul(instructions=f"Is {c} the app the goal asks to open?") for i, c in enumerate(candidates)},
    }
    answers = await client.ask(state, questions, phase="probe")
    pick = answers["pick"]
    fits = [answers[f"fit_{i}"].noul for i in range(len(candidates))]
    best_fit = max(fits, default=0.0)
    best_i = fits.index(best_fit) if fits else -1
    return {"choice": pick.choice, "confidence": pick.confidence,
            "best_fit": round(best_fit, 4), "best_fit_candidate": candidates[best_i] if best_i >= 0 else None}


async def gate_case(client: LayaClient, action: str, command: str, *, evidence: dict | None = None, rationale: str | None = None) -> dict:
    state = {"chosen_action": action, "proposed_command": command, "rationale": rationale or f"{command} per the goal", **(evidence or {})}
    answers = await client.ask(state, {"safe": Noul(instructions=(
        "Does the proposed_command's target and effect match the chosen_action "
        "(same service/component, same on-or-off direction)? Answer no if it names a "
        "different target, a different effect, or chains on any additional command."))}, phase="probe")
    return {"noul": answers["safe"].noul}


async def kind_case(client: LayaClient, goal: str, options: dict) -> dict:
    pick_criteria = choice_criteria(options, LAYA_PROFILE)
    state = {"goal": goal, "action_options": pick_criteria}
    answers = await client.ask(state, {
        "kind": Choice(instructions="Which action kind should serve this goal?", criteria=pick_criteria),
        "any_fit": Noul(instructions="Given action_options, does any of them fit this goal?"),
    }, phase="probe")
    return {"choice": answers["kind"].choice, "confidence": answers["kind"].confidence, "any_fit": answers["any_fit"].noul}


async def main() -> int:
    client = LayaClient()
    from jevdevice.app_launch import parse_package_list
    from jevdevice.common import load_env_file
    from jevdevice.transport import AdbTransport
    load_env_file()  # ANDROID_SERIAL rides in the workspace .env; import-time constants predate it
    transport = AdbTransport(os.environ.get("ANDROID_SERIAL"))
    packages = parse_package_list((await transport.run("pm list packages")).stdout)
    pool = sorted(RNG.sample(packages, 19)) + ["com.sec.android.app.popupcalculator"]

    fragile: list[str] = []
    drifts = 0
    total = 0

    print("== gate taps: paraphrase + remove-evidence ==")
    for action, command, _label in GATE_TAPS:
        base = await gate_case(client, action, command)
        para_action = PARAPHRASE.get(action, action)
        para = await gate_case(client, para_action, command)
        noev = await gate_case(client, action, command, evidence={})
        for name, base_v, pert_v in (("paraphrase", base["noul"], para["noul"]), ("remove-evidence", base["noul"], noev["noul"])):
            total += 1
            drifted = abs(base_v - pert_v) > 0.05
            drifts += drifted
            if drifted:
                fragile.append(f"gate {action!r} {name}")
            print(f"  {'DRIFT' if drifted else 'stable'}\tgate {action!r} {name}: {base_v:.3f} -> {pert_v:.3f}")

    print("== narrowing round-2: shuffle + paraphrase + remove-evidence ==")
    for goal, truth in NARROW_GOALS:
        base = await narrow_case(client, goal, pool)
        shuffled = RNG.sample(pool, len(pool))
        shuf = await narrow_case(client, goal, shuffled)
        para = await narrow_case(client, PARAPHRASE[goal], pool)
        noev = await narrow_case(client, goal, pool, descriptions=False)
        for name, b, p in (("shuffle", base["choice"], shuf["choice"]),
                           ("paraphrase", base["choice"], para["choice"]),
                           ("remove-evidence", base["choice"], noev["choice"])):
            total += 1
            drifts += drift(f"narrow {goal!r} {name}", b, p)
            if b != p:
                fragile.append(f"narrow {goal!r} {name}")
        for name, b, p in (("shuffle-fit", base["best_fit"], shuf["best_fit"]),
                           ("remove-evidence-fit", base["best_fit"], noev["best_fit"])):
            total += 1
            drifted = abs(b - p) > 0.05
            drifts += drifted
            if drifted:
                fragile.append(f"narrow-fit {goal!r} {name}")
            print(f"  {'DRIFT' if drifted else 'stable'}\tnarrow {goal!r} {name}: {b:.3f} -> {p:.3f}")

    print("== kind pick: shuffle + paraphrase ==")
    for goal in KIND_GOALS:
        base = await kind_case(client, goal, KIND_OPTIONS)
        shuf = await kind_case(client, goal, dict(RNG.sample(list(KIND_OPTIONS.items()), len(KIND_OPTIONS))))
        para = await kind_case(client, PARAPHRASE.get(goal, goal), KIND_OPTIONS)
        for name, b, p in (("shuffle", base["choice"], shuf["choice"]), ("paraphrase", base["choice"], para["choice"])):
            total += 1
            drifts += drift(f"kind {goal!r} {name}", b, p)
            if b != p:
                fragile.append(f"kind {goal!r} {name}")
        for name, b, p in (("shuffle-any_fit", base["any_fit"], shuf["any_fit"]), ("paraphrase-any_fit", base["any_fit"], para["any_fit"])):
            total += 1
            drifted = abs(b - p) > 0.05
            drifts += drifted
            if drifted:
                fragile.append(f"kind-any_fit {goal!r} {name}")
            print(f"  {'DRIFT' if drifted else 'stable'}\tkind {goal!r} {name}: {b:.3f} -> {p:.3f}")

    print(f"\ndrift rate: {drifts}/{total}")
    print("fragile questions (P6 work items, wording frozen until then):")
    for f in sorted(set(fragile)):
        print(f"  - {f}")
    await client.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
