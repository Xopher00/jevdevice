"""The seeded-VM task family: read-only questions about a fixture-filled Linux
microVM (files, a small git repo, logs, stock-machine facts), through the SAME
gated-probe runner as the CLI family (`run_probe`, imported from cli_family --
zero duplication of the judge/gate spine).

Subcommands (all honor the workspace .env; never auto-approves):
  seed-check   no Jev calls: one sandbox, re-seeded before each dev goal; checks
               CORRECT's command against EXPECT (and warns if a distractor also matches)
  run          dev-half goals end-to-end: one sandbox, re-seeded per goal -> run_probe

Candidate sets, CORRECT and EXPECT below are FROZEN task-family data, same
shape as cli_family.py's CANDIDATE_COMMANDS: closed sets, no free-form command
generation. CORRECT/EXPECT are seed-check-only (never shown to the judge).

Process goals: the dev half has none (the one process question, vm18, is
held-out), so re-seeding only has to rebuild files.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from datetime import timedelta

PHASES = os.path.dirname(os.path.abspath(__file__))
if PHASES not in sys.path:
    sys.path.insert(0, PHASES)

from cli_family import run_probe
from splitguard import assert_dev_only, dev_goals

from jevdevice.common import bootstrap, load_env_file
from jevdevice.device import SandboxDevice, SandboxShellTransport

FAMILY_NAME = "jevdevice-vm"
SANDBOX_TIME_LIMIT = timedelta(minutes=30)  # matches device/sandbox.py's SANDBOX_TIME_LIMIT

# Seeded once under /tmp/vmfix (absolute path -- does not depend on $HOME).
# Deterministic: fixed file contents/sizes, fixed git identity/branch/messages.
# No network commands anywhere (the sandbox denies egress regardless).
SEED_SCRIPT = r"""
set -euo pipefail
rm -rf /tmp/vmfix
mkdir -p /tmp/vmfix/data /tmp/vmfix/logs /tmp/vmfix/app
cat > /tmp/vmfix/notes.txt <<'EOF'
vmfix bring-up notes
seed script ran at build time
check TODO for next step
end of notes
EOF
cat > /tmp/vmfix/TODO <<'EOF'
TODO: rotate the logs next
EOF
head -c 100 /dev/zero > /tmp/vmfix/data/a.bin
head -c 5000 /dev/zero > /tmp/vmfix/data/b.bin
head -c 1234 /dev/zero > /tmp/vmfix/data/c.bin
head -c 10 /dev/zero > /tmp/vmfix/data/d.bin
printf 'line one\nline two\n' > /tmp/vmfix/logs/app.log
printf 'error occurred\n' > /tmp/vmfix/logs/error.log
printf 'access ok\n' > /tmp/vmfix/logs/access.log
printf 'see logs\n' > /tmp/vmfix/logs/readme.txt
cd /tmp/vmfix/app
git init -q -b vmfix-main
git config user.name vmfix-seed
git config user.email vmfix-seed@example.invalid
cat > config.ini <<'EOF'
[general]
mode = maintenance
EOF
cat > run.sh <<'EOF'
#!/bin/bash
# vmfix run script
echo vmfix-run
EOF
chmod +x run.sh
export GIT_AUTHOR_DATE='2026-01-01T00:00:00'
export GIT_COMMITTER_DATE='2026-01-01T00:00:00'
git add config.ini
git commit -q -m 'Initial vmfix scaffold'
git add run.sh
git commit -q -m 'Add config and run script'
""".strip()

# --- frozen candidate sets (task-family data, closed vocabularies) ---------------

CANDIDATE_COMMANDS: dict[str, list[str]] = {
    "vm01": [  # lines in notes.txt
        "wc -l < /tmp/vmfix/notes.txt",
        "wc -l < /tmp/vmfix/app/config.ini",
        "wc -l < /tmp/vmfix/logs/app.log",
    ],
    "vm03": [  # git branch checked out
        "git -C /tmp/vmfix/app rev-parse --abbrev-ref HEAD",
        "git -C /tmp/vmfix/app log -1 --pretty=%s",
        "cat /tmp/vmfix/app/config.ini",
    ],
    "vm05": [  # latest commit message
        "git -C /tmp/vmfix/app log -1 --pretty=%s",
        "git -C /tmp/vmfix/app rev-parse --abbrev-ref HEAD",
        "git -C /tmp/vmfix/app log -1 --pretty=%an",
    ],
    "vm07": [  # is run.sh executable
        "test -x /tmp/vmfix/app/run.sh && echo yes || echo no",
        "test -x /tmp/vmfix/notes.txt && echo yes || echo no",
        "stat -c %A /tmp/vmfix/app/run.sh",
    ],
    "vm09": [  # python3 version
        "python3 --version",
        "bash --version | head -1",
        "python2 --version",
    ],
    "vm11": [  # CPU cores
        "nproc",
        "who",
        "free -m",
    ],
    "vm13": [  # mode in config.ini
        "grep '^mode' /tmp/vmfix/app/config.ini",
        "grep '^mode' /tmp/vmfix/notes.txt",
        "cat /tmp/vmfix/app/run.sh",
    ],
    "vm15": [  # .log files in logs
        "find /tmp/vmfix/logs -maxdepth 1 -name '*.log' | wc -l",
        "find /tmp/vmfix/data -maxdepth 1 -type f | wc -l",
        "ls /tmp/vmfix/logs | wc -l",
    ],
    "vm17": [  # lines in run.sh
        "wc -l < /tmp/vmfix/app/run.sh",
        "wc -l < /tmp/vmfix/notes.txt",
        "wc -l < /tmp/vmfix/app/config.ini",
    ],
    "vm19": [  # kernel version
        "uname -r",
        "uname -m",
        "cat /tmp/vmfix/notes.txt",
    ],
}

CORRECT: dict[str, str] = {
    "vm01": "wc -l < /tmp/vmfix/notes.txt",
    "vm03": "git -C /tmp/vmfix/app rev-parse --abbrev-ref HEAD",
    "vm05": "git -C /tmp/vmfix/app log -1 --pretty=%s",
    "vm07": "test -x /tmp/vmfix/app/run.sh && echo yes || echo no",
    "vm09": "python3 --version",
    "vm11": "nproc",
    "vm13": "grep '^mode' /tmp/vmfix/app/config.ini",
    "vm15": "find /tmp/vmfix/logs -maxdepth 1 -name '*.log' | wc -l",
    "vm17": "wc -l < /tmp/vmfix/app/run.sh",
    "vm19": "uname -r",
}

EXPECT: dict[str, str] = {
    "vm01": r"^4\s*$",
    "vm03": r"^vmfix-main\s*$",
    "vm05": r"^Add config and run script\s*$",
    "vm07": r"^yes\s*$",
    "vm09": r"^Python 3\.\d+\.\d+\s*$",
    "vm11": r"^\d+\s*$",
    "vm13": r"mode\s*=\s*maintenance",
    "vm15": r"^3\s*$",
    "vm17": r"^3\s*$",
    "vm19": r"^\d+\.\d+\S*\s*$",
}


def dev_vm_goals() -> list[dict]:
    """The dev half of the vm_questions family -- the held-out half is never
    exposed here, asserted through the shared split loader."""
    goals = dev_goals("vm_questions")
    assert_dev_only(goals)
    return goals


# --- sandbox plumbing (lazy vercel import, monkeypatchable for offline tests) ----


# One sandbox per run, re-seeded before each goal: destroyed sandboxes keep counting toward
# the plan's concurrency cap (10 on Hobby) for minutes, so per-goal copies exhaust it.
RESEED_SCRIPT = "rm -rf /tmp/vmfix && " + SEED_SCRIPT


async def _new_sandbox(client):
    from vercel.sandbox import NetworkPolicy

    return await client.create_sandbox(network_policy=NetworkPolicy.deny_all(), execution_time_limit=SANDBOX_TIME_LIMIT)


async def seed_and_run(client, goals: list[dict], per_goal) -> list:
    """One sandbox for the whole run: re-seed the fixture before each goal, then
    per_goal(goal, device). The sandbox is destroyed whatever per_goal raises."""
    sandbox = await _new_sandbox(client)
    try:
        device = SandboxDevice(SandboxShellTransport(sandbox))
        results = []
        for goal in goals:
            seeded = await sandbox.run_process("bash", ["-lc", RESEED_SCRIPT], capture_output=True)
            if seeded.returncode != 0:
                raise RuntimeError(f"seed script failed (exit {seeded.returncode}): {seeded.stderr}")
            results.append(await per_goal(goal, device))
        return results
    finally:
        await sandbox.destroy()


# --- seed-check: zero Jev calls, checks CORRECT/EXPECT against the seed ----------


async def _seed_check_one(goal: dict, device) -> dict:
    gid = goal["id"]
    correct = CORRECT[gid]
    expect = EXPECT[gid]
    result = await device.run(correct)
    matched = bool(re.search(expect, result.stdout))
    ok = result.exit_code == 0 and matched
    print(f"{gid}: correct={correct!r} exit={result.exit_code} match={matched} ok={ok}")
    warnings = []
    for candidate in CANDIDATE_COMMANDS[gid]:
        if candidate == correct:
            continue
        distractor_result = await device.run(candidate)
        if re.search(expect, distractor_result.stdout):
            warnings.append(candidate)
            print(f"    WARNING: distractor also matches EXPECT (oracle too weak): {candidate!r}")
    return {"id": gid, "ok": ok, "distractor_warnings": warnings}


async def cmd_seed_check() -> int:
    client = _make_client()
    try:
        results = await seed_and_run(client, dev_vm_goals(), _seed_check_one)
    finally:
        await client.aclose()
    failures = [r["id"] for r in results if not r["ok"]]
    print(f"\nchecked: {len(results)}, failed: {len(failures)}")
    if failures:
        print(f"CORRECT command failed its EXPECT for: {failures}")
        return 1
    return 0


# --- run: per dev goal, a fresh sandbox copy through the gated-probe runner ------


async def cmd_run(goal_id: str | None) -> int:
    jev, _adb = bootstrap()  # adb precondition satisfied by the workspace .env; the adb device is unused here
    goals = dev_vm_goals()
    if goal_id:
        goals = [g for g in goals if g["id"] == goal_id]
        if not goals:
            print(f"no dev-half goal {goal_id!r}")
            return 2

    async def per_goal(goal: dict, device) -> dict:
        response = await run_probe(jev, device, goal["goal"], CANDIDATE_COMMANDS[goal["id"]], expect=EXPECT[goal["id"]])
        print(f"{goal['id']}: {response.get('status')} command={response.get('command')!r} "
              f"oracle={response.get('oracle')} satisfied={response.get('satisfied')}")
        return response

    client = _make_client()
    try:
        results = await seed_and_run(client, goals, per_goal)
    finally:
        await client.aclose()
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nruns: {len(results)}, ok: {ok}, not-ok: {len(results) - ok}")
    return 0


def _make_client():
    from vercel.sandbox import SandboxClient

    return SandboxClient.create()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seed-check", "run"))
    parser.add_argument("--goal", help="run one goal id (default: every dev-half task)")
    args = parser.parse_args()
    # Pick the artifact before any ask (see cli_family.py); setdefault only, no shadow traffic by default.
    os.environ.setdefault("JEV_QUESTION_SET", "v2")
    os.environ.setdefault("JEV_SHADOW", "0")
    load_env_file()  # VERCEL_TOKEN/TEAM_ID/PROJECT_ID must be set before the client is created
    if args.command == "seed-check":
        return asyncio.run(cmd_seed_check())
    return asyncio.run(cmd_run(args.goal))


if __name__ == "__main__":
    raise SystemExit(main())
