"""Live smoke: a few dev goals through the MCP tool surface with
JEV_ENGINE=laya (no API key needed), then a journal replay of the rows those
goals produced. Dev-split goals only (eval/goals.yaml).

Run: ANDROID_SERIAL=<serial> JEV_ENGINE=laya uv run python eval/phases/smoke_laya.py
"""

from __future__ import annotations

import asyncio
import json
import time

from mcp.shared.memory import create_connected_server_and_client_session

# Dev goals from eval/goals.yaml (frozen 2026-09-19). Held-out half untouched.
GOALS = [
    "What is the battery level?",        # ph01, question -> dumpsys
    "Open the Calculator app.",          # pa01, action -> launch
    "Turn Bluetooth off.",               # pa03, action -> gated toggle
    "Turn Bluetooth on.",                # pa04, restore the toggle
]


def _payload(result) -> dict:
    if result.content and hasattr(result.content[0], "text"):
        try:
            return json.loads(result.content[0].text)
        except json.JSONDecodeError:
            return {"raw": result.content[0].text[:200]}
    return {"error": [str(c) for c in result.content]}


async def run_goals() -> list[dict]:
    from jevdevice.mcp_server import mcp

    rows = []
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        for goal in GOALS:
            t0 = time.perf_counter()
            result = await session.call_tool("device_do", {"goal": goal})
            elapsed = time.perf_counter() - t0
            rows.append({"goal": goal, "elapsed_s": round(elapsed, 1), "response": _payload(result)})
    return rows


def journal_rows_for_goals() -> list[dict]:
    from jevdevice.journal.decision_log import get_journal, goal_id_for

    ids = {goal_id_for(goal) for goal in GOALS}
    rows = list(get_journal().replay())
    verdicts = {r["call_id"]: r.get("status") for r in rows if r.get("type") == "verdict"}
    return [{
        "type": row["type"], "engine": row.get("engine"), "phase": row.get("phase"),
        "verification": verdicts.get(row.get("call_id")), "call_id": row.get("call_id"),
        "command": row.get("executed_command"), "error": row.get("error"),
        "answers": {k: {"choice": a.get("choice"), "noul": a.get("noul"), "confidence": a.get("confidence")}
                    for k, a in (row.get("answers") or {}).items()},
    } for row in rows if (row.get("goal_id") or (row.get("scope") or {}).get("goal_id")) in ids]


async def main() -> None:
    results = await run_goals()
    print("=== DEVICE RESULTS ===")
    for row in results:
        print(json.dumps(row))
    print("=== JOURNAL ROWS (this session's goals) ===")
    for row in journal_rows_for_goals():
        print(json.dumps(row))


if __name__ == "__main__":
    asyncio.run(main())
