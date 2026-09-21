"""P7.5 T4 driver: run the tiered planner over the dev PHONE goals, unattended.

P5/P7 operational lessons, all load-bearing here:
- one SUBPROCESS per goal with a hard timeout (a sweep fallback can cost minutes),
- resumable runs.jsonl (first result wins; timed-out runs re-run),
- JEV_SHADOW=0 forced (CPU laya adds ~20 s of background work per ask for nothing),
- the planner NEVER approves: needs_approval counts as an escalated step
  (fail-closed) and tiers fall through -- device_approve stays human,
- every goal is a dev-half goal (held-out asserted absent from the plan).

Usage: uv run python eval/phase75_run_goals.py run [--limit N]  /  report
Tier telemetry afterwards: uv run python eval/phase75_report.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import yaml

from jevdevice.decision_log import goal_id_for

FLYWHEEL_DIR = REPO / "eval" / "phase75_recipes"
RUNS_FILE = FLYWHEEL_DIR / "runs.jsonl"
GOAL_TIMEOUT_S = 600   # per-goal subprocess timeout (P5: fallbacks cost minutes)
SLEEP_BETWEEN_RUNS_S = 20  # rate limit: device-safe pacing


def dev_phone_goals() -> list[dict]:
    data = yaml.safe_load((REPO / "eval" / "goals.yaml").read_text())
    goals = [e for e in data["phone_questions"] if e.get("split") == "dev"]
    heldout = {goal_id_for(e["goal"]) for e in data["phone_questions"] if e.get("split") == "heldout"}
    assert not ({goal_id_for(g["goal"]) for g in goals} & heldout), "held-out goal in the dev plan"
    return goals


def _read_done() -> set[str]:
    if not RUNS_FILE.exists():
        return set()
    done = set()
    for line in RUNS_FILE.read_text().splitlines():
        if line.strip():
            run = json.loads(line)
            if run.get("status") == "ok":
                done.add(run["goal"])
    return done


def _append_run(run: dict) -> None:
    FLYWHEEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_FILE, "a") as handle:
        handle.write(json.dumps(run, default=str) + "\n")


def run_one(goal: str) -> int:
    """Worker subprocess: bootstrap the real stack and resolve one goal."""
    os.environ.setdefault("JEV_SHADOW", "0")
    os.environ.setdefault("TYPESAFE_AI_API", os.environ.get("TYPESAFE_AI_API", ""))

    from jevdevice import planner
    from jevdevice.common import bootstrap

    async def main() -> dict:
        jev, transport = bootstrap()
        started = time.monotonic()
        result = await planner.resolve(jev, transport, goal, verbose=False)
        return {
            "goal": goal,
            "goal_id": goal_id_for(goal),
            "tier": result.tier,
            "status": result.status,
            "tier_path": result.tier_path,
            "recipe_id": result.recipe_id,
            "steps": [vars(step) for step in result.steps],
            "wall_s": round(time.monotonic() - started, 1),
            "usage": str(jev.usage.snapshot()),
        }

    out = asyncio.run(main())
    print(json.dumps(out, default=str))
    return 0


def cmd_run(limit: int | None) -> int:
    done = _read_done()
    goals = [g["goal"] for g in dev_phone_goals() if g["goal"] not in done]
    if limit is not None:
        goals = goals[:limit]
    print(f"runs planned: {len(goals)} (resuming; {len(done)} already ok)")
    for goal in goals:
        print(f"--- {goal}")
        started = time.monotonic()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "run-one", goal],
            capture_output=True, text=True, timeout=GOAL_TIMEOUT_S, check=False,
        )
        wall = round(time.monotonic() - started, 1)
        entry = {"goal": goal, "goal_id": goal_id_for(goal), "status": "ok" if proc.returncode == 0 else "worker_error",
                 "wall_s": wall, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
        _append_run(entry)
        time.sleep(SLEEP_BETWEEN_RUNS_S)
    print("batch complete")
    return 0


def cmd_report() -> int:
    runs = [json.loads(line) for line in RUNS_FILE.read_text().splitlines() if line.strip()]
    tiers: dict = {}
    for run in runs:
        if run.get("status") != "ok":
            continue
        payload = json.loads(run["stdout_tail"].strip().splitlines()[-1]) if run.get("stdout_tail") else {}
        tier = payload.get("tier")
        tiers.setdefault(str(tier), {"runs": 0, "resolved": 0})
        tiers[str(tier)]["runs"] += 1
        tiers[str(tier)]["resolved"] += 1 if payload.get("status") == "resolved" else 0
    summary = {"runs": len(runs), "by_tier": tiers}
    FLYWHEEL_DIR.mkdir(parents=True, exist_ok=True)
    (FLYWHEEL_DIR / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("usage: phase75_run_goals.py run [--limit N] | run-one '<goal>' | report", file=sys.stderr)
        return 2
    if args[0] == "run":
        limit = int(args[2]) if len(args) > 2 and args[1] == "--limit" else None
        return cmd_run(limit)
    if args[0] == "run-one":
        return run_one(args[1])
    if args[0] == "report":
        return cmd_report()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
