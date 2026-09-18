"""Test 1: prove interrupt()/Command(resume=...) works for our real gate,
against the real phone, before building the full graph around it.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from jevdevice.gate import GateVerdict, gate_command
from jevdevice.jev import JevClient
from jevdevice.gate import CommandVariant
from jevdevice.transport import AdbTransport

SERIAL = "192.168.100.11:46585"


class State(TypedDict):
    command: str
    chosen_label: str
    gate_verdict: str
    gate_reason: str
    approved: bool
    exit_code: int


async def gate_check(state: State) -> dict:
    api_key = os.environ["OPENROUTER_API_KEY"]
    jev = JevClient(api_key)
    result = await gate_command(
        jev, CommandVariant(command=state["command"], rationale="test"),
        chosen_label=state["chosen_label"], threshold=0.995,  # absurdly high: force needs_approval
    )
    print(f"gate_check: verdict={result.verdict} reason={result.reason} noul={result.noul_confidence}")
    return {"gate_verdict": result.verdict, "gate_reason": result.reason}


def route_after_gate(state: State) -> str:
    if state["gate_verdict"] == GateVerdict.DENIED:
        return END
    if state["gate_verdict"] == GateVerdict.NEEDS_APPROVAL:
        return "await_approval"
    return "execute"


def await_approval(state: State) -> dict:
    print("await_approval: pausing via interrupt() ...")
    decision = interrupt({"command": state["command"], "reason": state["gate_reason"]})
    print(f"await_approval: resumed with decision={decision}")
    return {"approved": decision}


def route_after_approval(state: State) -> str:
    return "execute" if state.get("approved") else END


async def execute(state: State) -> dict:
    transport = AdbTransport(SERIAL)
    result = await transport.run(state["command"])
    print(f"execute: ran {state['command']!r}, exit_code={result.exit_code}")
    return {"exit_code": result.exit_code}


def build_graph():
    graph = (
        StateGraph(State)
        .add_node("gate_check", gate_check)
        .add_node("await_approval", await_approval)
        .add_node("execute", execute)
        .add_edge(START, "gate_check")
        .add_conditional_edges("gate_check", route_after_gate, ["await_approval", "execute", END])
        .add_conditional_edges("await_approval", route_after_approval, ["execute", END])
        .add_edge("execute", END)
        .compile(checkpointer=InMemorySaver())
    )
    return graph


async def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set")

    graph = build_graph()
    config = {"configurable": {"thread_id": "test-interrupt-1"}}

    print("=== first invoke (expect pause) ===")
    result = await graph.ainvoke(
        {"command": "svc bluetooth disable", "chosen_label": "disable bluetooth"}, config,
    )
    print(f"result: {result}\n")
    if "__interrupt__" not in result:
        print("FAIL: expected an interrupt, got none")
        return
    print(f"interrupt payload: {result['__interrupt__']}\n")

    print("=== resume with approval ===")
    result = await graph.ainvoke(Command(resume=True), config)
    print(f"result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
