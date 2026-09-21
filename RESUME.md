# RESUME

Read the last entries of LOGBOOK.md first. Full execution protocol: `agent-plans/README.md`.

## Where things stand (2026-09-21)

The primary interface is the MCP server (`jevdevice-mcp`, 3 tools: `device_do`,
`device_screenshot`, `device_approve`). `device_do` picks ONE atomic action kind
per call via the judge, runs it gated, and journals everything. Code changes to
`.py` files need a client-side MCP reconnect to take effect.

## Module map (`src/jevdevice/`)

- `transport.py` — adb + uiautomator2 (frozen).
- `jev.py` — the hosted judge client; the frozen ask() wire contract lives here.
- `laya_backend.py` — in-process judge twin behind `JEV_ENGINE=jev|laya`.
- `budget.py` — per-engine budget/calibration knobs (`current_profile(engine)`).
- `question_sets/` — the frozen question phrasings (v1.yaml); runtime wording
  overrides are journaled, never anonymous.
- `narrowing.py`, `matching.py`, `ui.py`, `services.py`, `app_launch.py`,
  `elements.py` — candidate narrowing, gating, and the per-kind actions.
- `gate.py` — deny-list + one safety Noul per mutating command (no labeler —
  the old "small-model labeler" premise is corrected in phase-08-audits.md).
- `dispatch.py` — kind table, `pick_kind`, `run_kind` (the ONE shared
  propose→gate→execute path), `response_for` (per-kind response builders).
- `mcp_server.py` — the MCP surface; thin over dispatch + outcomes.
- `outcomes.py` — shared outcome-row emission; `graph_edge` bracketing
  (`JEV_GRAPH_EDGE` knob, telemetry-only).
- `decision_log.py` — append-only JSONL journal under `~/.jevdevice/journal/`.
- `recipes.py`, `planner.py` — stored verified chains (recipes) and the tiered
  resolver (recipe hit → bounded adapt → stepwise selection → hand-back → human).
- `shadow.py` — the second-engine observer (`JEV_SHADOW=0` off).
- `calibrate/` — calibration CLIs + the continuous calibration loop.
- `common.py` — bootstrap() and the confidence-gate helper.

## Eval tooling

Isolated under `eval/phases/` with functional names (see its README):
calibration/refit harnesses, the perturbation batch runner, recovery mining,
fine-tune export/train/eval, recipe build, planner batch + tier report,
plus `splitguard.py` (the one shared dev/held-out split loader — held-out goals
are sacred; nothing tunes, trains, or exports on them).

## Standing rules

One LOGBOOK entry per phase; held-out half sacred; gates stay fail-closed;
frozen files stay frozen (transport.py, the ask() wire shape, matching.py
verification); every new constant is a named knob.
