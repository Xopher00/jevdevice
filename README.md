# jevdevice

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

jevdevice turns a plain-language goal into exactly one verified action on a real device.

## Why

Scripted device automation uses a fixed map: exact coordinates, resource IDs, one decision tree
per app. This map breaks the moment an app changes its layout, and it never covers an app nobody
tested against. jevdevice reads the real device state before every action, so its choices adjust
when the app changes. A typed judge model picks only from those real candidates, and every choice
runs through a safety gate before execution.

## Install

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

The local-shell device family needs nothing more. The Android phone family also needs `adb`, a
phone reachable over USB or wireless debugging, and `ANDROID_SERIAL` set (from `adb devices`).
The default hosted judge engine needs a `TYPESAFE_AI_API` key. Put both in a `.env` file one
directory above the repo, or at the workspace root. jevdevice reads it automatically.

`uv sync` also fetches [typesymbolic](https://github.com/Xopher00/typesymbolic) directly from
GitHub, pinned to the revision in `pyproject.toml`'s `[tool.uv.sources]`. See "The typesymbolic
dependency" below for what it provides.

## Usage

```sh
uv run python eval/phases/cli_family.py run   # dev goal set, local shell only, no phone or key
```

```text
=== pc17: Which users are currently logged in?
picked pick: who (confidence 0.92, any_fit 0.91)
gate verdict: needs_approval (jev_uncertain, noul=0.65)
    -> escalated command=None satisfied=None
runs: 10, ok: 4, not-ok: 6
```

With a phone connected:

```sh
uv run python -m jevdevice.execution.dispatch "open the calculator"   # one goal
uv run python demo.py                                                 # guided interactive demo
```

## Status

Active development, no fixed release. The phone and local-shell device families, the safety gate,
and the decision journal work today with the hosted `jev` engine. The in-process `laya` engine
swaps in without code changes but does not yet reach `jev`'s accuracy. Run it only on CUDA, not
on CPU. The calibration loop and recipe recall are new, and confirmed so far only on small live
samples.

## Tests

```sh
uv run pytest --deselect tests/test_mcp_approval_flow.py   # 257 offline tests, no device or network
uv run ruff check                                           # lint, must stay clean
```

`test_mcp_approval_flow.py` is the one live integration test. It needs a connected phone and a
real API key.

## Contributing

Issues and pull requests are welcome. For larger changes, open an issue first to discuss the
approach.

## License

[MIT](LICENSE) © 2026

---

## Architecture

### The core idea

Most automation tools work from a hand-written map: exact coordinates, resource IDs, a fixed
decision tree per app. That map breaks when an app updates, and it never covers an app the author
did not test. jevdevice hand-writes nothing device-specific. Every action follows the same
pipeline.

1. **Enumerate** the real, current device state.
2. **Narrow** the real candidate list to one, with the judge engine.
3. **Gate** the resulting command: a deny-list, then one direct safety judgment from the judge.
   Anything uncertain needs a human decision instead of running.
4. **Execute** and **verify** against device truth (exit code, live screen state, real command
   output).

No step can select something that does not exist on the device at that moment, so the action set
is correct by construction. The judge is only ever asked typed questions over closed option sets,
so it can be calibrated, replayed, and swapped. That is how the engine below got exchanged for a
second one without a single call site changing.

### The typesymbolic dependency

jevdevice runs on [typesymbolic](https://github.com/Xopher00/typesymbolic), a separate repo that
supplies the judge engine, question primitives, decision journal, gate types, and calibration
math everything above rests on. jevdevice consumes it as a pinned git revision
(`pyproject.toml`'s `[tool.uv.sources]`) and never modifies it. A core change this repo needs
becomes a change in typesymbolic itself, not an edit here.

What jevdevice takes from typesymbolic:

- Question primitives: `Noul`, `Choice`, `Score`, `Question`, `QuestionRef`, `Answer`, re-exported
  unchanged from `jev.py`.
- The engine and transport: `JevEngine`, `ask_batch()`.
- The journal: append-only decision and outcome rows, replay, and blob storage for large values.
- Gate types: `GateVerdict`, `GateResult`, `circuit.threshold_decision()`.
- Calibration: `current_threshold()`, `recalibrate()` (tighten-only unless a human signs off).
- Planner episodes: one `Episode` per `resolve()` call.

What stays domain-owned in jevdevice, with no typesymbolic counterpart: candidate narrowing
(`matching.py`), per-engine budget and threshold profiles (`budget.py`), usage accounting
(`ledger.py`), two-round narrowing and shadow-engine plumbing (`judge/narrowing.py`,
`judge/shadow.py`), and the `question_sets` wording surface.

### Judge engines

One judge contract, two interchangeable implementations, selected by `JEV_ENGINE`:

- **`jev`** (default): TypeSafe's hosted System One model
  ([docs.typesafe.ai](https://docs.typesafe.ai)). Needs `TYPESAFE_AI_API`. About 0.3 seconds per
  judgment.
- **`laya`**: an in-process checkpoint, pinned to an exact revision, that answers the identical
  question contract locally (`asyncio.to_thread` predict). Needs no API key. `JEV_DEVICE` selects
  CPU or CUDA, and `JEV_LAYA_REVISION` pins the checkpoint (the default is the calibrated base
  revision).

Both engines see the same frozen question wordings (below), record usage into a shared ledger,
and journal every decision. Runs from either engine are comparable, replayable, and safe to A/B.

### Device families and the Device protocol

Device knowledge sits behind a five-member protocol (`src/jevdevice/device/protocol.py`): `name`
(journal identity), `dump_hierarchy()` (the snapshot), `run()` (one shell command, exit code and
stdout as device truth), `run_binary()` (raw stdout bytes), `window_size()` (geometry). Engine
modules type against the protocol only.

- **`AdbDevice`**: the Android phone, over a frozen adb/uiautomator2 transport. The snapshot is
  the live accessibility-tree XML. A warm read costs about 0.3 seconds because the uiautomator2
  companion stays resident (a plain `uiautomator dump` costs about 2.2 seconds per call in process
  startup alone).
- **`CliDevice`**: the local shell. The snapshot is plain text (`pwd` plus a listing), so the
  engine's XML parser rejects it and screen-grounded paths fail closed instead of assuming a
  screen exists. The CLI family's real path is the gated-command spine, with a frozen candidate
  command set per goal. It needs no phone at all, which makes it useful for exercising,
  calibrating, and training the judge on pure-text device data.

Adding a family means writing one adapter. The engine itself stays untouched. The second family
was chosen on purpose to break the protocol's phone-shaped assumptions (screen geometry, "the
snapshot is a UI tree") rather than confirm them.

### What the judge is asked, and who writes it

Judge questions are frozen, versioned artifacts, never prose improvised at runtime:
`src/jevdevice/question_sets/v1.yaml` (phone family) and `v2.yaml` (adds the local-shell family's
wordings, and keeps every v1 entry byte-identical). `JEV_QUESTION_SET` selects the version. Call sites
read their question text from the artifact. An unknown question id or a missing slot raises an
error instead of falling back to improvised wording. Runtime-generated wording exists only as a
journaled escape hatch, and can later be promoted into the next compiled version.

A compile validator (`eval/phases/compile_questions.py`) checks each frozen set against the
journal: every real-path question instance must match a template exactly, and the validator stamps
witness counts back into the artifact as provenance.

### Gates and safety

Every command that changes device state passes the gate before it runs:

- A **deny-list** rejects dangerous command shapes outright (`rm`, `reboot`, factory-wipe words,
  app uninstall or clear), and a **read-only classification** lets pure reads through without a
  judgment.
- Then **one safety judgment** from the engine ("does this command do exactly what the chosen
  action says, and nothing more?") runs against a calibrated confidence threshold.

Below the threshold, the verdict is `needs_approval` and the command does not run. The MCP
approval flow resolves it in a second tool call. The CLI prompts the terminal. Unattended runners
(the planner, the eval harnesses) never auto-approve, and an approval-pending verdict counts as a
failure. Gate thresholds are per-engine calibration knobs. Automatic recalibration may only ever
tighten them. Loosening a threshold needs recorded human sign-off.

### The decision journal

Everything is journaled, append-only, outside the repo (`~/.jevdevice/tsjournal/` by default,
`JEV_JOURNAL_DIR` overrides):

- **Decision rows** record every judge ask: the full state, the questions, the answers, usage,
  latency, engine, and checkpoint revision.
- **Outcome rows** record every execution: what ran, the verification result (verified, failed,
  unverified, or escalated), the device identity, and a before/after foreground edge for
  trajectory building.
- Decision and outcome rows join by `call_id`, so an approval, its judgment, and its execution
  replay as one chain. Large values live in a content-addressed blob store.

The journal is the project's data flywheel. Calibration pulls distributions from it with zero
model calls, the fine-tune exporter builds training sets from verified rows, the recipe builder
aggregates verified chains, and the question compiler validates wordings against it. Every outcome
row carries its device family's name, so data from different device families stays separate by
construction.

### Recipes and the tiered planner

Multi-step goals resolve through a tiered planner (`src/jevdevice/execution/planner.py`) that
never generates a free-form plan.

1. A **stored recipe** (a previously verified chain of actions for the same or a similar goal)
   runs first. The judge only fills variable slots.
2. A stored recipe that mostly fits gets **adapted**: each step is checked against the live device,
   drop-only.
3. Otherwise the planner **selects steps itself**, choosing among the engine's closed action
   vocabulary with a done-check after each step.
4. A goal nothing covers goes back to the **calling agent** to break down. Each step still runs
   gated through the same engine.
5. An unresolved goal escalates to a **human**.

Every tier transition is journaled. Verified runs feed back into the recipe store automatically.

### Calibration

Each engine has its own budget and threshold profile (`src/jevdevice/budget.py`): option-count
limits, state truncation, chunk sizes, abstain options, and the gate and confidence thresholds,
all named knobs, never bare numbers. The calibration CLIs (`src/jevdevice/calibrate/`) refit
profiles from journal data. A continuous rolling-window loop refits quality metrics (Brier score,
expected calibration error, precision) from stored rows with zero model calls, and proposes
threshold changes that run in shadow mode before any tighten-only promotion.

### The MCP surface (three tools, kept minimal)

- `device_do(goal, verify, auto_approve)`: resolves one goal to one atomic action (kind selection,
  propose, gate, execute, verify). Sequencing multi-step tasks stays the calling agent's job.
- `device_approve(thread_id, decision)`: resolves a pending `needs_approval` action. A denied
  command never runs, in any mode.
- `device_screenshot()`: returns the live screen.

### Eval tooling and the data split

`eval/phases/` holds the batch and analysis tooling (functional names, see its own README): A/B
runners, shadow-agreement reports, threshold refits, gate audits, question compilation, the
fine-tune export and Kaggle training script, a perturbation harness, recovery-pair mining,
training-format exporters, recipe building, and METR Task Standard-shaped adapters over both
device families. `eval/goals.yaml` holds the plain-language goal sets with a frozen dev/held-out
split. The held-out half is never run, tuned on, or exported, and every batch tool checks this
through one shared loader.

### Why this design

- **Ground everything in device truth.** Candidates, verification, and escalation all come from
  what the device actually reports.
- **Keep the judge's job small and typed.** Closed option sets, frozen wordings, and calibrated
  thresholds make an engine swappable and its decisions auditable.
- **Fail closed.** When confidence is short, the action does not run. A human decides, or the goal
  escalates.
- **Journal everything.** The journal is the single source of labeled data, provenance, and replay
  for every later improvement: calibration, fine-tunes, recipes, question compilation.
