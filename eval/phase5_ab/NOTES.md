# Phase 5 run notes — shadow mode + dev-half A/B (2026-09-20/21)

## Runs

| file | what |
|---|---|
| `ab-jev.jsonl` | jev primary, all 13 dev phone goals, shadow ON (default) — 10 ok / 2 escalated / 2... see scorecard |
| `ab-laya.jsonl` | laya primary, first 5 goals (900 s timeout hit mid-run) |
| `ab-laya-rest.jsonl` | per-goal resumed laya attempts — aborted by operator decision, no further laya runs |
| `scorecard.json` | merged A/B scorecard (M1 template) |
| `agreement-report.json` | T2 report over the journal (69 shadow rows, all paired) |

## Findings

1. **Shadow mechanism works.** 69/69 shadow rows paired to their jev primaries
   via `shadow_of`; zero shadow errors surfaced into the primary path; jev
   latency unaffected (p50 334 ms / p95 738 ms — after-mode guarantees this
   structurally).
2. **No shadow ANSWERS were captured: laya on CPU is ~20 s/ask** (laya arm:
   judge ask p50 20.2 s, p95 70.2 s). The jev run finished in ~40 s while
   shadow asks were still mid-predict; the shutdown burst cancelled all 69,
   each emitting a request-only row (`answers=null, error=null`). Agreement
   n=0 this run. To capture real shadow answers: run the shadow with
   `JEV_DEVICE=cuda` (the single attached checkpoint fits the 4 GiB card —
   P4 ran narrowing on CUDA; the OOM was the three-checkpoint Router preload)
   or accept that CPU laya can only shadow long-lived sessions.
3. **A/B (strict "status==ok" rule, escalations count as unattended failure):**
   - jev: 0.692 success (9.5/13 → 9 ok, ph03/pa01 escalated by jev's own gate,
     ph05/ph07 ran-but-unverified dumpsys), 5.7 judge calls/goal, p50 2.7 s /
     p95 4.1 s per goal.
   - laya: 0.000 success on 5/13 completed goals (every one escalated at
     `pick_kind`'s any_fit gate — matches P2/P4), 14.4 judge calls/goal,
     p50 11.2 s / p95 461 s per goal (the 461 s = chunked-sweep fallback on
     CPU). The remaining 8 goals were not run: with a ~20 s/ask engine that
     escalates everything, the outcome is not in doubt and each sweep-trigger
     goal costs minutes. **The laya arm was aborted deliberately** (operator
     call, logged).
4. **Flip decision: KEEP jev default.** Laya loses on every dev-half axis we
   measure (success 0.0 vs 0.692; latency 20 s vs 0.33 s per ask), on top of
   P4/P4.5's precision-bar failure. Reassess after P6's fine-tune; any future
   laya evaluation on this machine must use `JEV_DEVICE=cuda` or budget CPU
   minutes per ask.
