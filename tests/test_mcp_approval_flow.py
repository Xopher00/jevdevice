"""Integration test: drives mcp_server.py through a real MCP ClientSession (in-memory
transport, no subprocess/stdio needed) to prove the approval boundary is a real
request/response pair, not a blocking prompt disguised as one.
"""

from __future__ import annotations

import json

from mcp.shared.memory import create_connected_server_and_client_session

from jevdevice.actions.elements import dump_screen, parse_actionable_elements

# bootstrap() requires the TYPESAFE_AI_API key only for the jev engine, and
# test_decision_log's module import (which runs first in a full-suite collection)
# provides one for that case -- no placeholder needed here.
from jevdevice.common import SERIAL
from jevdevice.mcp_server import mcp
from jevdevice.transport import AdbTransport


def _payload(result) -> dict:
    return json.loads(result.content[0].text)


async def _formula_field() -> str | None:
    transport = AdbTransport(SERIAL)
    for label in parse_actionable_elements(await dump_screen(transport)):
        if "calc_edt_formula" in label:
            return label
    return None


# Needs a real connected Android device (adb) at SERIAL; not runnable without one.
async def test_mcp_approval_flow() -> None:
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        result = await session.call_tool("device_do", {"goal": "turn on bluetooth", "auto_approve": True})
        _payload(result)

        await session.call_tool("device_do", {"goal": "open the calculator"})
        before = await _formula_field()
        result = await session.call_tool("device_do", {"goal": "tap the 9 button"})
        payload = _payload(result)
        assert payload["status"] == "needs_approval", f"expected needs_approval, got {payload}"
        thread_id = payload["thread_id"]

        after_pending = await _formula_field()
        assert after_pending == before, "action must not execute before approval"

        result = await session.call_tool("device_approve", {"thread_id": thread_id, "decision": "approve"})
        _payload(result)

        after_approve = await _formula_field()
        assert after_approve != before, "approving should have let the tap through"

        result = await session.call_tool("device_approve", {"thread_id": thread_id, "decision": "approve"})
        payload = _payload(result)
        assert payload["status"] == "error", "a consumed thread_id must not be approvable again"
