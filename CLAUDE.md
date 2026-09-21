# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

jevdevice resolves a plain-language goal ("open the calculator", "turn Bluetooth off")
to exactly **one** real, verified action on a target device, or declines to act. A typed
judge engine (`jev`, TypeSafe's hosted System One model, or `laya`, an in-process
checkpoint) answers small closed-option questions — pick a candidate, is this command
safe, did it work — never free-form plans or commands. Candidates always come from live
device state, so the action set is correct by construction. Two device families today:
`AdbDevice` (Android phone over adb) and `CliDevice` (local shell, no phone needed).

Read `README.md` before making structural changes — it documents the full pipeline
(judge engines, Device protocol, question sets, gates, journal, planner/recipes,
calibration, MCP surface) in more depth than is worth repeating here.

## Commands

```sh
uv sync                                                  # install deps
uv run pytest                                            # ~190 offline tests (no device/network)
uv run pytest tests/test_gate.py                         # single file
uv run pytest tests/test_gate.py::test_name -v           # single test
uv run ruff check                                        # lint (must stay clean)
uv run jevdevice-mcp                                     # start the MCP server
uv run python -m jevdevice.dispatch "open the calculator"  # one goal, phone
uv run python eval/phases/cli_family.py run              # dev goals, local shell (no phone/API key)
```

`tests/test_mcp_approval_flow.py` is the one live integration test — needs a connected
phone (`adb devices`) and `TYPESAFE_AI_API`. Everything else in `tests/` is offline
(pure functions, scripted judges, recorded journals) and safe to run anytime.
`experiment/` is excluded from pytest collection (`norecursedirs` in `pyproject.toml`);
it's a separate sandbox, not part of the package.

A `.env` one directory above the repo, or at the workspace root, supplies
`TYPESAFE_AI_API` / `ANDROID_SERIAL` and is picked up automatically.

## Architecture

### Package layout (`src/jevdevice/`)

- `device/` — the 5-member `Device` protocol (`protocol.py`) plus `adb.py`
  (`AdbDevice`) and `cli.py` (`CliDevice`). Engine and action code type against the
  protocol only; adding a device family means writing one adapter here.
- `actions/` — per-kind propose/execute pairs (`app_launch`, `elements`, `services`,
  `ui`). Each action returns real candidates from a live snapshot; nothing is
  hard-coded about what exists on a device.
- `execution/dispatch.py` — `KIND_TABLE` normalizes every action kind's
  propose/execute into one shape, shared by the CLI (`run_toolkit`) and MCP
  (`mcp_server.py`'s `device_do`/`device_approve`) so they can't independently drift.
  Resolves exactly one goal to one kind; sequencing is always the caller's job.
- `execution/planner.py` — the tiered multi-step planner: stored recipe → adapted
  recipe → planner-selected steps → hand back to the calling agent → escalate to a
  human. Every tier transition is journaled.
- `execution/recipes.py` — the recipe store that `planner.py` reads/writes.
- `judge/gate.py` — deny-list + read-only classification + one safety judgment
  against a calibrated confidence threshold. Below threshold → `needs_approval`;
  unattended runners never auto-approve.
- `judge/narrowing.py`, `judge/shadow.py` — candidate narrowing via the judge, and
  shadow-engine comparison plumbing.
- `jev.py` — the hosted TypeSafe System One client (`noul`/`choice`/`score`
  primitives), pinned to a specific model version (not `-latest`) because
  `gate.py`'s threshold is calibrated against that version's confidence computation.
- `laya_backend.py` — the in-process checkpoint engine, same question contract as
  `jev.py`, selected via `JEV_ENGINE=laya`.
- `question_sets/` — frozen, versioned YAML wordings (`v1.yaml` phone, `v2.yaml`
  +CLI family). Call sites source question text from here; an unknown question id or
  missing slot raises rather than falling back to improvised wording.
  `JEV_QUESTION_SET` selects the version; `eval/phases/compile_questions.py` validates
  a set against the journal and stamps witness counts back in as provenance.
- `journal/` — append-only decision/outcome logging, written outside the repo to
  `~/.jevdevice/journal/`. Decision rows and outcome rows join by `call_id`. This is
  the sole source of data for calibration, fine-tune export, and recipe building.
- `budget.py` — per-engine named knobs (option-count limits, truncation, chunk sizes,
  gate/confidence thresholds) — never bare numbers at call sites.
- `calibrate/` — refits budget/gate profiles from journal data; `continuous.py` is the
  rolling-window loop that proposes **tighten-only** threshold changes (loosening
  needs recorded human sign-off), shadow-run before promotion.
- `ledger.py` — usage/cost tracking shared across engines.
- `mcp_server.py` — the three-tool MCP surface: `device_do`, `device_approve`,
  `device_screenshot`.

### Engine swap points

Everything upstream of `jev.py`/`laya_backend.py` (dispatch, actions, gate, planner)
types against the same question-in/answer-out contract. `JEV_ENGINE` picks the
implementation; `JEV_DEVICE` (cpu/cuda) and `JEV_LAYA_REVISION` (pinned checkpoint)
configure `laya` specifically. Default engine is `jev` — **laya on CPU is known bad**
(~20s/ask, 0% success in the dev-half A/B); only run laya with `JEV_DEVICE=cuda`.

### Eval tooling (`eval/`)

`eval/phases/` holds batch/analysis scripts, isolated from the package (functional
names — see `eval/phases/README.md` for the historical phase-number mapping).
`eval/phases/splitguard.py` is the **one shared loader** enforcing the frozen
dev/held-out split in `eval/goals.yaml`; the held-out half must never be run, tuned
on, or exported — every batch tool asserts this through that loader. Local run output
directories (`recalibration/`, `ab/`, `finetune/`, `flywheel/`, `recipes/`, `cli/`)
are regenerated, untracked, and hold machine-local data (serials, hostnames);
`flywheel/TRAINING_FORMAT.md` is the tracked exception.

### Project docs, in reading order for resuming work

- `RESUME.md` — current state, what's unblocked/pending, do-not-repeat notes (e.g. the
  laya-on-CPU result above). Read this first when picking work back up.
- `LOGBOOK.md` — one entry per phase, decision-by-decision design history (untracked
  in spirit but present locally; large).
- `agent-plans/README.md` — the execution protocol for phase work; `agent-plans/phase-*.md`
  carry per-phase handoff notes.
- `PLAN.md` — founding design rationale.

## Conventions specific to this repo

- Judge questions are never improvised at call sites — add new wordings to a
  question-set YAML and bump `JEV_QUESTION_SET`, don't inline strings.
- Gate/budget numbers are named constants in `budget.py`, not literals at call sites.
- Any new device family implements the `Device` protocol in full; don't special-case
  a family inside `actions/` or `execution/dispatch.py`.
- Automatic threshold recalibration is tighten-only; a loosening change requires a
  recorded human sign-off, not just a passing shadow run.
