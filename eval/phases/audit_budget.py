"""Reducer audit: does any ask() state exceed the answering engine's budget
profile without a logged truncation? Reads stored journal rows only -- no
model calls. Run after a device session (a day of rows makes this meaningful;
a handful of smoke rows is still a real check of the plumbing).

Run: uv run python eval/phases/audit_budget.py [journal_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jevdevice.budget import PROFILES, current_profile
from jevdevice.journal.decision_log import DecisionJournal


def _json_bytes(value) -> int:
    return len(json.dumps(value, default=str).encode("utf-8"))


def main() -> int:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".jevdevice" / "journal"
    rows = [r for r in DecisionJournal(directory).replay() if r.get("type") == "decision"]
    if not rows:
        print("no decision rows found")
        return 1

    print("engine profiles: " + ", ".join(
        f"{p.engine} max_len={p.max_len} head={p.head_max_len} screen={p.screen_limit} chunk={p.chunk_size} probe={p.probe_max_chars}"
        for p in PROFILES.values()
    ))
    print(f"rows: {len(rows)}\n")
    print("engine\tphase\tstate_bytes\ttokens_total\tquestions\t~tokens_per_q\ttruncated\tviolates_budget")
    worst: dict[tuple, dict] = {}
    violations = 0
    for row in rows:
        engine = row.get("engine") or "jev"
        profile = current_profile(engine)
        state_bytes = _json_bytes(row.get("state"))
        usage = row.get("usage") or {}
        input_tokens = usage.get("input_tokens") or 0
        # The in-process engine reports the per-question SUM: each question gets
        # its own window under max_len, so the per-question share is what can
        # violate the budget. This machine's calibration point: 912 state bytes
        # -> 827 tokens on one question.
        n_questions = max(1, len(row.get("questions") or {}))
        tokens_per_q = input_tokens / n_questions if input_tokens else int(state_bytes * 827 / 912)
        truncated = bool(row.get("truncation"))
        error = row.get("error")
        violates = tokens_per_q > profile.max_len and not truncated and not error
        violations += violates
        key = (engine, row.get("phase"))
        if tokens_per_q > worst.get(key, {}).get("tokens_per_q", -1):
            worst[key] = {"tokens_per_q": int(tokens_per_q), "state_bytes": state_bytes, "n_questions": n_questions, "total": input_tokens}
        print(f"{engine}\t{row.get('phase')}\t{state_bytes}\t{input_tokens}\t{n_questions}\t{int(tokens_per_q)}\t{truncated}\t{violates}")

    print("\nworst per (engine, phase):")
    for (engine, phase), w in sorted(worst.items()):
        limit = current_profile(engine).max_len
        headroom = limit - w["tokens_per_q"]
        print(f"{engine}\t{phase}\t~{w['tokens_per_q']} tokens/question of {limit} (headroom {headroom}; {w['n_questions']} questions, {w['total']} total; state {w['state_bytes']} bytes)")
    print(f"\nviolations (per-question over budget, no truncation logged, no error): {violations}")
    return 0 if violations == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
