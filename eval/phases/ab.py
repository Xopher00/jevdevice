"""Dev-half A/B through both engines: success rate, judge calls/goal,
p50/p95 latency per goal. Dev-split goals only (eval/goals.yaml); the
held-out half stays untouched until the engine decision is final.

Two modes (a subprocess per engine, because the engine client is constructed
at import of mcp_server and one process can only ever be one engine):

  uv run python eval/phases/ab.py run --engine jev   # on-device worker; JSONL to stdout
  uv run python eval/phases/ab.py run --engine laya
  uv run python eval/phases/ab.py scorecard jev.jsonl laya.jsonl   # offline, journal-only

Worker success rule (unattended): status "ok" == success. Anything else —
"escalated", "needs_approval", "unverified", "error" — is a failure, because
an unattended caller cannot approve through the human gate and gates stay
fail-closed. The harness NEVER calls device_approve.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def dev_phone_goals() -> list[dict]:
    """Dev-half, phone-family goals (this device is a phone; pc_* belongs to
    the PC family). Guarded: a held-out id reaching this list is a hard error."""
    import splitguard

    goals = [g for family in ("phone_questions", "phone_actions")
             for g in splitguard.dev_goals(family)]
    splitguard.assert_dev_only(goals)
    return goals


async def run_worker(engine: str, only: list[str] | None = None) -> None:
    from jevdevice.common import load_env_file

    load_env_file()  # before anything reads os.environ (common.SERIAL is read at import)
    assert os.environ.get("JEV_ENGINE", "jev") == engine, "parent must set JEV_ENGINE"

    from mcp.shared.memory import create_connected_server_and_client_session

    from jevdevice.journal.decision_log import goal_id_for
    from jevdevice.mcp_server import mcp

    specs = dev_phone_goals()
    if only:
        wanted = set(only)
        specs = [g for g in specs if g["id"] in wanted]
        assert len(specs) == len(wanted), "unknown goal id in --ids"
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        for spec in dev_phone_goals():
            t0 = time.perf_counter()
            result = await session.call_tool("device_do", {"goal": spec["goal"]})
            wall_s = round(time.perf_counter() - t0, 2)
            payload = {}
            if result.content and hasattr(result.content[0], "text"):
                try:
                    payload = json.loads(result.content[0].text)
                except json.JSONDecodeError:
                    payload = {"status": "error", "raw": result.content[0].text[:200]}
            print(json.dumps({
                "engine": engine, "id": spec["id"], "goal": spec["goal"],
                "goal_id": goal_id_for(spec["goal"]), "wall_s": wall_s,
                "status": payload.get("status"), "reasons": payload.get("reasons"),
            }), flush=True)


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, round(p * (len(values) - 1)))], 2)


def scorecard(worker_files: list[Path]) -> dict:
    from jevdevice.journal import decision_log

    rows = list(decision_log.get_journal().replay())
    primaries = [r for r in rows if r.get("type") == "decision" and not r["scope"].get("shadow_of")]

    by_engine: dict[str, list[dict]] = {}
    for path in worker_files:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):  # tolerate loop noise / shell echoes in resumed batches
                continue
            run = json.loads(line)
            seen = {r["id"] for r in by_engine.setdefault(run["engine"], [])}
            if run["id"] not in seen:  # resumed batches may repeat ids; first result wins
                by_engine[run["engine"]].append(run)

    cards = {}
    for engine, runs in by_engine.items():
        goal_ids = {r["goal_id"] for r in runs}
        decisions = [r for r in primaries if r["scope"].get("goal_id") in goal_ids and r["engine"] == engine]
        per_goal: dict[str, int] = {}
        for r in decisions:
            per_goal[r["goal_id"]] = per_goal.get(r["goal_id"], 0) + 1
        successes = sum(1 for r in runs if r["status"] == "ok")
        elapsed = [r["elapsed_ms"] for r in decisions if isinstance(r.get("elapsed_ms"), (int, float))]
        incomplete = [r["id"] for r in runs if r["status"] == "timeout"]
        cards[engine] = {
            "goals": len(runs),
            "success_rate": round(successes / len(runs), 3) if runs else None,
            "incomplete_goals": incomplete,
            "failures": {r["id"]: {"status": r["status"], "reasons": r["reasons"]}
                         for r in runs if r["status"] != "ok"},
            "judge_calls_per_goal": round(statistics.fmean(
                per_goal.get(r["goal_id"], 0) for r in runs), 1) if runs else None,
            "goal_wall_s": {"p50": _pct([r["wall_s"] for r in runs], 0.5),
                            "p95": _pct([r["wall_s"] for r in runs], 0.95)},
            "judge_ask_ms": {"p50": _pct(elapsed, 0.5), "p95": _pct(elapsed, 0.95),
                             "n": len(elapsed)},
        }
    return cards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "scorecard"])
    parser.add_argument("--engine", choices=["jev", "laya"], default="jev")
    parser.add_argument("--ids", nargs="*", default=None, help="run only these goal ids (resumable)")
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    if args.mode == "run":
        asyncio.run(run_worker(args.engine, args.ids))
    else:
        if not args.files:
            raise SystemExit("scorecard needs the worker JSONL files")
        print(json.dumps(scorecard(args.files), indent=1))


if __name__ == "__main__":
    main()
