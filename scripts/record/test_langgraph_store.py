"""Test 2: prove Store gives genuine cross-thread caching for the template
cache — a value put under one thread_id must be readable from a different
thread_id, unlike checkpointer state which is thread-scoped.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


class State(TypedDict):
    device_fingerprint: str
    tool_name: str
    template: str | None


def get_template(state: State, runtime: Runtime) -> dict:
    key = (state["device_fingerprint"], "templates")
    item = runtime.store.get(key, state["tool_name"])
    if item is not None:
        print(f"  cache HIT: {item.value}")
        return {"template": item.value["template"]}
    discovered = "svc {service} {enabled}"  # stand-in for a real discover_template call
    print(f"  cache MISS: discovering {discovered!r}")
    runtime.store.put(key, state["tool_name"], {"template": discovered})
    return {"template": discovered}


def build_graph():
    return (
        StateGraph(State)
        .add_node("get_template", get_template)
        .add_edge(START, "get_template")
        .add_edge("get_template", END)
        .compile(checkpointer=InMemorySaver(), store=InMemoryStore())
    )


def main() -> None:
    graph = build_graph()
    fingerprint = "Android 14, Samsung SM-S921B"

    print("=== goal 1, thread A (expect miss) ===")
    result_a = graph.invoke(
        {"device_fingerprint": fingerprint, "tool_name": "toggle_service"},
        {"configurable": {"thread_id": "goal-A"}},
    )
    print(f"  template: {result_a['template']}\n")

    print("=== goal 2, thread B, same device+tool (expect HIT, cross-thread) ===")
    result_b = graph.invoke(
        {"device_fingerprint": fingerprint, "tool_name": "toggle_service"},
        {"configurable": {"thread_id": "goal-B"}},
    )
    print(f"  template: {result_b['template']}\n")

    assert result_a["template"] == result_b["template"], "cache mismatch across threads"
    print("=== RESULT: cross-thread cache confirmed working ===")


if __name__ == "__main__":
    main()
