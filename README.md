# jevdevice

jevdevice is a device-automation harness with a minimal MCP tool surface. An
LLM agent gives it a goal in plain language — "open the calculator", "turn
Bluetooth off", "what is the kernel version" — and the harness resolves it to
exactly **one** real, verified action on the target device, or declines to
act. It runs today against two device families: an **Android phone** (over
adb) and the **local shell** (any Linux/macOS box, no extra hardware).

Two things make it different from a scripted automator:

1. **Nothing is hard-coded about what's on the device.** Every decision picks
   from candidates discovered live (the on-screen accessibility tree, the
   installed app list, a service dump, a directory listing). If the app
   updates its layout tomorrow, the candidates change with it.
2. **A typed judge engine decides; deterministic code executes.** The judge
   never emits free-form plans or commands — it answers small, typed
   questions (pick one of these real candidates; is this command safe; did
   the action achieve the goal). Everything it picks is then gated before it
   runs, and everything that runs is journaled.

## The core idea

Most automation tools work from a hand-written map: exact coordinates,
resource IDs, a fixed decision tree per app. That breaks when an app updates
and never generalizes to apps the author didn't test. jevdevice hand-writes
nothing device-specific. Every action follows the same pipeline:

1. **Enumerate** the real, current device state.
2. **Narrow** the real candidate list to one, with the judge engine.
3. **Gate** the resulting command: a deny-list, then one direct safety
   judgment from the judge. Anything uncertain needs a human decision
   instead of running.
4. **Execute** and **verify** against device truth (exit code, live screen
   state, real command output).

No step can select something that does not exist on the device at that
moment, so the action set is correct by construction. And because the judge
is only ever asked typed questions over closed option sets, it can be
calibrated, replayed, and swapped — which is how the engine below got
exchanged for a second one without touching call sites.

## Judge engines

One judge contract, two interchangeable implementations, selected by
`JEV_ENGINE`:

- **`jev`** (default) — TypeSafe's hosted System One model
  ([docs.typesafe.ai](https://docs.typesafe.ai)). Needs `TYPESAFE_AI_API`.
  ~0.3 s per judgment.
- **`laya`** — an in-process checkpoint, pinned to an exact revision, that
  answers the identical question contract locally (`asyncio.to_thread`
  predict). Needs no API key; `JEV_DEVICE` selects cpu/cuda, and
  `JEV_LAYA_REVISION` pins the checkpoint (the default is the calibrated
  base revision).

Both engines see the same frozen question wordings (below), emit usage into
a shared ledger, and journal every decision — so runs are comparable,
replayable, and safe to A/B.

## Device families and the Device protocol

Device knowledge sits behind a five-member protocol (`src/jevdevice/device/protocol.py`):
`name` (journal identity), `dump_hierarchy()` (the snapshot), `run()`
(one shell command — exit code and stdout are device truth), `run_binary()`
(raw stdout bytes), `window_size()` (geometry). Engine modules type against
the protocol only.

- **`AdbDevice`** — the Android phone, wrapping a frozen adb/uiautomator2
  transport. The snapshot is the live accessibility-tree XML; a warm read
  costs ~0.3 s because the uiautomator2 companion stays resident (a naive
  `uiautomator dump` costs ~2.2 s per call in process startup alone).
- **`CliDevice`** — the local shell. The snapshot is honest plain text
  (`pwd` + listing), so the engine's XML parser rejects it and screen-grounded
  paths degrade fail-closed instead of pretending there's a screen; the CLI
  family's real path is the gated-command spine, with frozen candidate
  command sets per goal. It needs no phone at all — useful for exercising,
  calibrating and training the judge on pure-text device data.

Adding a family means writing one adapter; the engine is untouched. The
second family was deliberately chosen to break the protocol's phone-shaped
assumptions (screen geometry, "the snapshot is a UI tree") rather than
flatter them.

## What the judge is asked, and who writes it

Judge questions are **frozen, versioned artifacts**, never prose improvised
at runtime: `src/jevdevice/question_sets/v1.yaml` (phone family) and
`v2.yaml` (adds the local-shell family's wordings; every v1 entry
byte-identical). `JEV_QUESTION_SET` selects the version. Call sites source
their question text from the artifact; an unknown question id or a missing
slot raises instead of falling back to improvisation. Runtime-generated
wording exists only as a journaled escape hatch, promotable into the next
compiled version.

A compile validator (`eval/phases/compile_questions.py`) proves each frozen
set against the journal: every real-path question instance must match a
template exactly, and witness counts are stamped back into the artifact as
provenance.

## Gates and safety

Every command that changes device state passes the gate before it runs:

- a **deny-list** rejects dangerous command shapes outright (`rm`, `reboot`,
  factory-wipe words, app uninstall/clear, ...), and a **read-only
  classification** lets pure reads through without a judgment;
- then **one safety judgment** from the engine ("does this command do exactly
  what the chosen action says, and nothing more?") against a calibrated
  confidence threshold.

Below the threshold the verdict is `needs_approval` and the command does not
run. The MCP approval flow resolves it in a second tool call; the CLI prompts
the terminal; unattended runners (the planner, the eval harnesses) **never
auto-approve** — an approval-pending verdict counts as a failure. Gate
thresholds are per-engine calibration knobs; automatic recalibration may
only ever **tighten** them (loosening requires recorded human sign-off).

## The decision journal

Everything is journaled, append-only, outside the repo
(`~/.jevdevice/journal/`):

- **decision rows** — every judge ask: the full state, the questions, the
  answers, usage, latency, engine + checkpoint revision;
- **outcome rows** — every execution: what ran, the verification result
  (verified / failed / unverified / escalated), the device identity, and a
  before/after foreground edge for trajectory building;
- joined by `call_id`, so an approval, its judgment, and its execution
  replay as one chain. Large values live in a content-addressed blob store.

The journal is the project's data flywheel: calibration pulls distributions
from it (zero model calls), the fine-tune exporter builds training sets from
verified rows, the recipe builder aggregates verified chains, and the
question compiler validates wordings against it. Every outcome row carries
the device's name, so data from different device families separates by
construction.

## Recipes and the tiered planner

Multi-step goals resolve through a tiered planner (`src/jevdevice/execution/planner.py`)
that never generates free-form plans:

1. a **stored recipe** (a previously verified chain of actions for the same
   or a similar goal) is tried first — the judge only fills variable slots;
2. a stored recipe that mostly fits is **adapted**, step-checked against the
   live device, drop-only;
3. otherwise the planner **selects steps itself**, choosing among the
   engine's closed action vocabulary with a done-check after each;
4. a goal nothing covers is handed back to the **calling agent** to
   decompose (each step still runs gated through the same engine), and
5. unresolved goals escalate to a **human**.

Every tier transition is journaled. Verified runs are aggregated back into
the recipe store automatically.

## Calibration

Each engine has its own budget/threshold profile (`src/jevdevice/budget.py`):
option-count limits, state truncation, chunk sizes, abstain options, and the
gate/confidence thresholds — all named knobs, never bare numbers. Profiles
are refit from journal data by the calibration CLIs
(`src/jevdevice/calibrate/`), and a continuous rolling-window loop re-fits
quality metrics (Brier, ECE, precision) from stored rows with **zero model
calls**, proposing threshold changes that are shadow-run before any
tighten-only promotion.

## The MCP surface (three tools, kept minimal)

- `device_do(goal, verify, auto_approve)` — resolves ONE goal to ONE atomic
  action (kind selection → propose → gate → execute → verify). Sequencing
  multi-step tasks is the calling agent's job.
- `device_approve(thread_id, decision)` — resolves a pending
  `needs_approval` action; a denied command never runs, in any mode.
- `device_screenshot()` — the live screen.

## Eval tooling and the data split

`eval/phases/` holds the batch/analysis tooling (functional names, see its
README): A/B runners, shadow-agreement reports, threshold refits, gate
audits, question compilation, the fine-tune export + Kaggle training
script, a perturbation harness, recovery-pair mining, training-format
exporters, recipe building, and METR Task Standard-shaped adapters over both
device families. `eval/goals.yaml` holds the plain-language goal sets with a
**frozen dev/held-out split**; the held-out half is never run, tuned on, or
exported, and every batch tool asserts this through one shared loader.

## Setup

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/).

For the phone family: `adb` with an Android phone reachable over USB or
wireless debugging, `ANDROID_SERIAL` set (from `adb devices`), and — for the
default hosted engine — `TYPESAFE_AI_API`. A `.env` one directory above the
repo or at the workspace root is picked up automatically.

```sh
uv sync                       # install
adb devices                   # confirm the phone is visible
uv run jevdevice-mcp           # start the MCP server
```

The local-shell family needs neither a phone nor an API key path beyond the
engine's own requirements:

```sh
uv run python -m jevdevice.execution.dispatch "open the calculator"   # one goal, phone, CLI
uv run python eval/phases/cli_family.py run                # dev goals, local shell
```

## Tests

`uv run pytest` runs ~190 offline tests (pure functions, scripted judges,
recorded journals — no device, no network). `tests/test_mcp_approval_flow.py`
is the one live integration test: it needs a connected phone and a real API
key. `uv run ruff check` keeps the code lint-clean.

## Why this design (the short version)

- **Ground everything in device truth** — candidates, verification, and
  escalation all come from what the device actually reports.
- **Keep the judge's job small and typed** — closed option sets, frozen
  wordings, calibrated thresholds: that's what makes an engine swappable and
  its decisions auditable.
- **Fail closed** — when confidence is short, the action doesn't run; a
  human decides or it escalates.
- **Journal everything** — the journal is the single source of labeled data,
  provenance, and replay for every later improvement (calibration,
  fine-tunes, recipes, question compilation).

## License

MIT. See `LICENSE`.
