"""Replay-with-perturbation harness for trajectory collection.

Warm paths generate no data (cache hits skip the judge) and a static goal list
overfits the exact wording it was verified on, so every verified run here is a
trajectory and every failure is a recovery-pair candidate:

  - mutated goal wording (meaning-preserving wrappers; question phrasings stay
    compiled -- the GOAL is the input, not the judge's questions)
  - different starting apps / pre-broken states (each setup is itself a DEV goal
    run through the same gated device_do path -- the harness executes nothing
    outside the gate, ever)
  - one subprocess per goal with a hard timeout (a single sweep goal can
    cost minutes; an unattended batch must never hang on one run)

Unattended rules (load-bearing): status "ok" == success, escalations are
failures, device_approve is NEVER called, gates stay fail-closed, shadow off.

Dev-half data only: every goal text is asserted absent from the held-out half.

  uv run python eval/phases/perturbation_harness.py plan
  uv run python eval/phases/perturbation_harness.py run [--limit N] [--ids id ...]
  uv run python eval/phases/perturbation_harness.py run-one --goal "..." [--variant v]
  uv run python eval/phases/perturbation_harness.py report
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# The dev-goal loader lives in ab.py (single source of truth, reused).
from ab import dev_phone_goals as p5_dev_phone_goals

FLYWHEEL_DIR = REPO / "eval" / "phases" / "flywheel"
RUNS_FILE = FLYWHEEL_DIR / "runs.jsonl"
TRAJECTORIES_FILE = FLYWHEEL_DIR / "trajectories.jsonl"

# --- named knobs ----------------------------------------------------------------
GOAL_TIMEOUT_S = 600          # per-run subprocess timeout (sweep fallbacks cost minutes)
SLEEP_BETWEEN_RUNS_S = 20     # rate limit: device-safe pacing between runs
BATCH_SIZE = None             # max runs per `run` invocation (None = the whole plan)
HELDOUT_GUARD_CASEFOLD = True # asserted on every plan build

# Meaning-preserving goal-wording mutations. A str template slots {goal}; a
# callable takes the goal text. Nothing here may change WHAT the goal asks
# (verify() and the judge's compiled questions both key off the same intent).
WORDING_MUTATIONS: dict[str, object] = {
    "as_is": "{goal}",
    "polite": "Please — {goal}",
    "quick": "Quick one: {goal}",
    "lowercase": str.lower,
}

# Start states: each maps to DEV goals run first through device_do (gated,
# journaled under their own goal_id) so the main goal starts from that state.
START_STATES: dict[str, list[str]] = {
    # goal ids into eval/goals.yaml (dev half only)
    "home": [],
    "calculator": ["pa01"],
    "camera": ["pa08"],
}

# Pre-broken states: setups that deliberately start a goal from the state it
# is supposed to change (a recovery-pair generator for T2). Key = goal id,
# value = setup goal ids run before it.
BROKEN_STATE_SETUPS: dict[str, list[str]] = {
    "pa04": ["pa03"],  # Turn Bluetooth on, starting from Bluetooth-off
    "pa03": ["pa04"],  # Turn Bluetooth off, starting from Bluetooth-on
    "pa08": ["pa01"],  # Open Camera, starting from the Calculator
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def apply_mutation(goal_text: str, mutation_id: str) -> str:
    """One meaning-preserving wording mutation, or ValueError on an unknown id."""
    try:
        mutation = WORDING_MUTATIONS[mutation_id]
    except KeyError:
        raise ValueError(f"unknown mutation id {mutation_id!r} (knows: {sorted(WORDING_MUTATIONS)})") from None
    return mutation(goal_text) if callable(mutation) else mutation.format(goal=goal_text)


def build_plan() -> list[dict]:
    """The run matrix: per dev phone goal, a baseline + a wording mutation +
    either a pre-broken-state variant (when one is defined for the goal) or a
    different-start-state variant. Deterministic: the mutation and start-state
    choices key off the goal id's hash, not wall-clock randomness."""
    import yaml

    dev_specs = p5_dev_phone_goals()
    raw = yaml.safe_load((REPO / "eval" / "goals.yaml").read_text())
    heldout = {e["goal"].casefold() for section in ("pc_questions", "phone_questions", "phone_actions")
               for e in raw.get(section, []) if e["split"] == "heldout"}
    goal_texts_by_id = {e["id"]: e["goal"] for section in ("pc_questions", "phone_questions", "phone_actions")
                        for e in raw.get(section, [])}

    plan: list[dict] = []
    for spec in dev_specs:
        goal_text = spec["goal"]
        assert goal_text.casefold() not in heldout, f"held-out goal {spec['id']} reached the plan"
        mutations = [m for m in WORDING_MUTATIONS if m != "as_is"]
        goal_hash = int(hashlib.sha256(spec["id"].encode()).hexdigest(), 16)
        mutated_id = mutations[goal_hash % len(mutations)]
        variants = [
            {"variant_id": "baseline", "mutation_id": "as_is", "start_state": "home", "setup_ids": []},
            {"variant_id": f"wording_{mutated_id}", "mutation_id": mutated_id, "start_state": "home", "setup_ids": []},
        ]
        if spec["id"] in BROKEN_STATE_SETUPS:
            variants.append({"variant_id": "pre_broken", "mutation_id": "as_is",
                             "start_state": "pre_broken", "setup_ids": BROKEN_STATE_SETUPS[spec["id"]]})
        else:
            # Different starting app: a start state whose setup does not already
            # achieve the goal itself ("Open the Calculator" never starts FROM
            # the Calculator).
            state_ids = sorted(START_STATES)
            start_state = state_ids[goal_hash % len(state_ids)]
            setup_ids = [sid for sid in START_STATES[start_state] if sid != spec["id"]]
            conflicts = any(goal_texts_by_id.get(sid) == goal_text for sid in setup_ids)
            if setup_ids and not conflicts:
                variants.append({"variant_id": f"start_{start_state}", "mutation_id": "as_is",
                                 "start_state": start_state, "setup_ids": setup_ids})
        for variant in variants:
            mutated_goal = apply_mutation(goal_text, variant["mutation_id"])
            if mutated_goal == goal_text and variant["mutation_id"] != "as_is":
                continue  # a no-op mutation for this goal would double-journal the baseline
            plan.append({
                "run_id": f"{spec['id']}-{variant['variant_id']}",
                "goal_id": hashlib.sha256(mutated_goal.encode()).hexdigest()[:12],
                "goal": mutated_goal,
                "source_goal_id": spec["id"],
                "source_goal_text": goal_text,
                "variant_id": variant["variant_id"],
                "mutation_id": variant["mutation_id"],
                "start_state": variant["start_state"],
                "setup_ids": variant["setup_ids"],
            })
    return plan


def _load_done_run_ids() -> set[str]:
    if not RUNS_FILE.exists():
        return set()
    done: set[str] = set()
    for line in RUNS_FILE.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        run = json.loads(line)
        if run.get("status") != "timeout":  # a timed-out run is re-run on resume
            done.add(run["run_id"])
    return done


def _append_run(run: dict) -> None:
    FLYWHEEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(run, separators=(",", ":"), default=str) + "\n")


# --- worker (one goal, one subprocess) ------------------------------------------

async def run_one(goal: str, setup_ids: list[str]) -> dict:
    """Run setups then the goal through the real gated MCP surface in-process.
    NEVER approves: auto_approve stays False, escalations are failures."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from jevdevice.common import load_env_file

    load_env_file()  # before anything reads os.environ (common.SERIAL is read at import)
    assert os.environ.get("JEV_SHADOW", "0") == "0", "shadow must be off for harness runs"
    assert os.environ.get("JEV_ENGINE", "jev") == "jev", "the harness drives the recorded default engine"

    from jevdevice.journal.decision_log import goal_id_for
    from jevdevice.mcp_server import mcp

    setup_statuses: list[dict] = []
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        for setup_id in setup_ids:
            specs = {s["id"]: s for s in p5_dev_phone_goals()}
            assert setup_id in specs, f"setup {setup_id} is not a dev phone goal"
            result = await session.call_tool("device_do", {"goal": specs[setup_id]["goal"], "auto_approve": False})
            payload = _payload_of(result)
            setup_statuses.append({"setup_id": setup_id, "status": payload.get("status")})
            if payload.get("status") != "ok":
                return {"status": "setup_failed", "setups": setup_statuses, "reasons": payload.get("reasons")}
        t0 = time.perf_counter()
        result = await session.call_tool("device_do", {"goal": goal, "auto_approve": False})
        wall_s = round(time.perf_counter() - t0, 2)
        payload = _payload_of(result)
    return {
        "status": payload.get("status"),
        "reasons": payload.get("reasons"),
        "wall_s": wall_s,
        "goal_id": goal_id_for(goal),
        "setups": setup_statuses,
        "never_approved": True,
    }


def _payload_of(result) -> dict:
    if result.content and hasattr(result.content[0], "text"):
        try:
            return json.loads(result.content[0].text)
        except json.JSONDecodeError:
            return {"status": "error", "reasons": [result.content[0].text[:200]]}
    return {"status": "error", "reasons": ["empty tool response"]}


def cmd_run(limit: int | None, wanted_ids: list[str]) -> int:
    plan = build_plan()
    if wanted_ids:
        plan = [r for r in plan if r["source_goal_id"] in set(wanted_ids) or r["run_id"] in set(wanted_ids)]
    done = _load_done_run_ids()
    queue = [r for r in plan if r["run_id"] not in done]
    if limit is not None:
        queue = queue[:limit]
    print(json.dumps({"planned": len(plan), "already_done": len(plan) - len(queue), "to_run": len(queue),
                      "knobs": {"GOAL_TIMEOUT_S": GOAL_TIMEOUT_S, "SLEEP_BETWEEN_RUNS_S": SLEEP_BETWEEN_RUNS_S}}))
    for index, item in enumerate(queue, start=1):
        started = _now_iso()
        child = subprocess.run(
            [sys.executable, str(Path(__file__)), "run-one", "--goal", item["goal"], "--setups", *item["setup_ids"]],
            capture_output=True, text=True, timeout=GOAL_TIMEOUT_S, check=False,
            env={**os.environ, "JEV_SHADOW": "0", "JEV_ENGINE": "jev"},
        )
        worker = {}
        for line in reversed((child.stdout or "").splitlines()):
            if line.strip().startswith("{"):
                worker = json.loads(line)
                break
        record = {
            **item,
            "started_at": started,
            "finished_at": _now_iso(),
            "status": worker.get("status") or ("timeout" if child.returncode != 0 else "error"),
            "reasons": worker.get("reasons"),
            "wall_s": worker.get("wall_s"),
            "setups": worker.get("setups"),
            "returncode": child.returncode,
        }
        _append_run(record)
        print(json.dumps({"i": index, "run_id": record["run_id"], "status": record["status"],
                          "wall_s": record["wall_s"]}), flush=True)
        time.sleep(SLEEP_BETWEEN_RUNS_S)
    return 0


def _ts_key(value: str) -> datetime:
    """Normalize a timestamp for window comparison, in the journal's naive-local
    frame. The runs index writes tz-aware UTC (`datetime.now(UTC).isoformat()`),
    the journal writes naive local time (`datetime.now().isoformat()`) -- aware
    stamps are converted to local and unwrapped so the join is
    convention-independent (and sentinel windows like year 0001 stay valid)."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed  # journal convention: naive local
    return parsed.astimezone().replace(tzinfo=None)  # runs index: aware UTC -> naive local


def _row_in_window(row: dict, started: datetime, finished: datetime) -> bool:
    ts = row.get("ts")
    if not ts:
        return False
    try:
        return started <= _ts_key(ts) <= finished
    except ValueError:  # unparseable stamp: never window-match
        return False


def _row_goal(row: dict) -> str | None:
    return (row.get("scope") or {}).get("goal") if row["type"] == "decision" else row.get("goal")


def extract_trajectories(journal, runs: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Pure core of `report`: journal-only join of each run's rows (primary only,
    ts-windowed) into trajectories + per-run status. Testable without a device."""
    rows = list(journal.replay())
    verdicts = {r["call_id"]: r["status"] for r in rows if r.get("type") == "verdict"}
    trajectories = []
    per_run_status = {}
    for run in runs:
        started, finished = _ts_key(run["started_at"]), _ts_key(run["finished_at"])
        window_rows = [
            row for row in rows
            if row.get("type") in ("decision", "outcome")
            and not (row.get("scope") or {}).get("shadow_of")
            and _row_goal(row) == run["goal"]
            and _row_in_window(row, started, finished)
        ]
        outcomes = [r for r in window_rows if r.get("type") == "outcome"]
        per_run_status[run["run_id"]] = {
            "worker_status": run["status"],
            "device_verified": any(verdicts.get(r.get("call_id")) == "verified" for r in outcomes),
            "n_decisions": sum(1 for r in window_rows if r.get("type") == "decision"),
            "n_outcomes": len(outcomes),
        }
        if window_rows:
            trajectories.append({
                "run_id": run["run_id"], "goal_id": run["goal_id"], "goal": run["goal"],
                "source_goal_id": run["source_goal_id"], "variant_id": run["variant_id"],
                "mutation_id": run["mutation_id"], "start_state": run["start_state"],
                "setup_ids": run["setup_ids"], "worker_status": run["status"],
                "rows": window_rows,
            })
    return trajectories, per_run_status


def cmd_report() -> dict:
    """Offline (journal-only) trajectory extraction + batch summary."""
    from jevdevice.journal.decision_log import get_journal

    journal = get_journal()
    runs = _read_runs()
    trajectories, per_run_status = extract_trajectories(journal, runs)
    FLYWHEEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAJECTORIES_FILE, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(t, separators=(",", ":"), default=str) + "\n" for t in trajectories)
    # worker status lives in the per-run rollup (extract_trajectories), not on
    # the raw runs.jsonl rows -- those carry the plain `status` field.
    statuses = [per_run_status[r["run_id"]]["worker_status"] for r in runs]
    summary = {
        "runs": len(runs),
        "verified_trajectories": sum(1 for s in per_run_status.values() if s["device_verified"]),
        "worker_ok": sum(1 for s in statuses if s == "ok"),
        "failures": sum(1 for s in statuses if s not in ("ok",)),
        "failure_kinds": {
            kind: statuses.count(kind)
            for kind in set(statuses) if kind != "ok"
        },
        "trajectory_rows_written": len(trajectories),
        "trajectory_file": str(TRAJECTORIES_FILE),
    }
    (FLYWHEEL_DIR / "batch_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def _read_runs() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    runs, seen = [], set()
    for line in RUNS_FILE.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        run = json.loads(line)
        if run["run_id"] not in seen:  # first result wins (resume semantics)
            seen.add(run["run_id"])
            runs.append(run)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "run", "run-one", "report"])
    parser.add_argument("--goal", help="run-one: the (already-mutated) goal text")
    parser.add_argument("--setups", nargs="*", default=[], help="run-one: setup goal ids")
    parser.add_argument("--limit", type=int, default=BATCH_SIZE)
    parser.add_argument("--ids", nargs="*", default=None)
    args = parser.parse_args()
    if args.mode == "plan":
        plan = build_plan()
        FLYWHEEL_DIR.mkdir(parents=True, exist_ok=True)
        (FLYWHEEL_DIR / "plan.json").write_text(json.dumps(plan, indent=2, default=str) + "\n")
        print(json.dumps({"planned_runs": len(plan),
                          "per_goal_variants": sorted({r["variant_id"] for r in plan})}))
        return 0
    if args.mode == "run":
        return cmd_run(args.limit, args.ids or [])
    if args.mode == "run-one":
        assert args.goal, "run-one needs --goal"
        result = asyncio.run(run_one(args.goal, args.setups))
        print(json.dumps(result, separators=(",", ":"), default=str), flush=True)
        return 0
    cmd_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
