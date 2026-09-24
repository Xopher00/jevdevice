"""Shadow agreement report (journal-only, zero model calls).

Pairs each shadow decision row (shadow_of set, engine=laya) with the primary
row it observed (call_id == shadow_of, engine=jev) and reports:
  - agreement overall and per phase (choice: winning option; noul: 0.5 cut,
    plus mean |delta|; abstentions reported separately)
  - latency comparison from the rows' own elapsed_ms (p50/p95 per engine)

Run: uv run python eval/phases/shadow_agreement.py
Output is pipe-friendly: JSON lines, then a human summary.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict

from jevdevice.journal import decision_log


def _choice_winner(answer: dict) -> str | None:
    """The option the distribution actually picks (confidence is a different
    quantity per engine -- the probabilities are the comparable thing)."""
    probabilities = answer.get("probabilities") or {}
    if not probabilities:
        return answer.get("choice")
    return max(probabilities, key=lambda k: probabilities[k])


def _compare(primary_answer: dict, shadow_answer: dict) -> dict:
    kind = primary_answer.get("type")
    if kind == "choice":
        p, s = _choice_winner(primary_answer), _choice_winner(shadow_answer)
        return {"kind": kind, "agree": p == s, "primary": p, "shadow": s,
                "abstain": s == "none_of_these"}
    if kind == "noul":
        p, s = primary_answer.get("noul"), shadow_answer.get("noul")
        return {"kind": kind, "agree": (p >= 0.5) == (s >= 0.5),
                "primary": p, "shadow": s, "abs_delta": round(abs(p - s), 4)}
    return {"kind": kind, "agree": None, "primary": primary_answer, "shadow": shadow_answer}


def build_report(rows: list[dict]) -> dict:
    primary_by_call = {r["call_id"]: r for r in rows
                       if r.get("type") == "decision" and not r["scope"].get("shadow_of")}
    pairs = [(primary_by_call[r["scope"]["shadow_of"]], r) for r in rows
             if r.get("type") == "decision" and r["scope"].get("shadow_of") in primary_by_call]

    per_phase: dict[str, list[dict]] = defaultdict(list)
    comparisons: list[dict] = []
    for primary, shadowed in pairs:
        for name, p_answer in (primary.get("answers") or {}).items():
            s_answer = (shadowed.get("answers") or {}).get(name)
            if s_answer is None:
                continue
            comp = _compare(p_answer, s_answer)
            comp.update({"phase": primary.get("phase"), "goal_id": primary["scope"].get("goal_id")})
            comparisons.append(comp)
            per_phase[comp["phase"]].append(comp)

    def _rate(comps: list[dict]) -> dict:
        scored = [c for c in comps if c["agree"] is not None]
        agrees = sum(1 for c in scored if c["agree"])
        return {
            "n": len(scored),
            "agreement": round(agrees / len(scored), 3) if scored else None,
            "abstentions": sum(1 for c in scored if c.get("abstain")),
            "mean_abs_noul_delta": round(statistics.fmean(
                c["abs_delta"] for c in scored if c["kind"] == "noul"), 4
            ) if any(c["kind"] == "noul" for c in scored) else None,
        }

    def _latency(engine: str) -> dict:
        values = [r["elapsed_ms"] for r in rows
                  if r.get("type") == "decision" and r.get("engine") == engine
                  and isinstance(r.get("elapsed_ms"), (int, float)) and r["elapsed_ms"] > 0]
        values.sort()
        if not values:
            return {"n": 0}
        def pct(p: float) -> float:
            idx = min(len(values) - 1, max(0, round(p * (len(values) - 1))))
            return values[idx]
        return {"n": len(values), "p50_ms": pct(0.5), "p95_ms": pct(0.95),
                "min_ms": values[0], "max_ms": values[-1]}

    shadow_rows = [r for r in rows if r.get("type") == "decision" and (r.get("scope") or {}).get("shadow_of")]
    shadowed_count = len(shadow_rows)
    primaries_with_shadow = len({r["scope"]["shadow_of"] for r in shadow_rows})
    return {
        "journal_rows": len(rows),
        "shadow_rows": shadowed_count,
        "primaries_shadowed": primaries_with_shadow,
        "overall": _rate(comparisons),
        "per_phase": {phase: _rate(comps) for phase, comps in sorted(per_phase.items())},
        "latency_ms": {"jev_primary": _latency("jev"), "laya_shadow": _latency("laya")},
        "comparisons": comparisons,
    }


def main() -> None:
    rows = list(decision_log.get_journal().replay())
    report = build_report(rows)
    for comp in report.pop("comparisons"):
        print(json.dumps(comp))
    print("=== SUMMARY ===")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
