# typesymbolic migration report — jevdevice (round 1)

**Date:** 2026-09-22 · **Scope:** migrate jevdevice onto `typesymbolic` as a
dependency, one subsystem at a time, replacing jevdevice's own code outright
(no adapter/bridge layers). Written for the agent developing typesymbolic:
what migrated cleanly, what was hard and why, and what core changes would
unblock the rest.

> **OPERATOR DECISION, 2026-09-22: typesymbolic is not jevdevice's to
> change — its agent owns it.** The migration below was implemented and
> live-verified end to end, then **fully reverted at operator direction**.
> Both repos are at their pre-migration state (typesymbolic at `7271aaf`,
> its 153 tests green; jevdevice 196/196 green including the live on-device
> test, no typesymbolic dependency). Everything in this file now stands as
> **tested specifications and findings for the typesymbolic agent** — the
> "Core changes" section describes implementations that passed both suites
> and the live device pipeline at the time, not landed code.

## ROUND 1 (re-landed 2026-09-22, operator-approved): dependency adoption, ZERO core changes

Scope fixed by the operator: adopt typesymbolic **as committed** (`7271aaf`);
extend in THIS repo only where core has the mechanism and the domain widens a
field or supplies a schema; no journal, no engine boundary. Everything below
is jevdevice-side code; typesymbolic's tree is untouched and its 153 tests
green at every step.

**What changed, and the why for each:**

1. **Dependency wiring.** `typesymbolic` as a uv path source, consumed as
   committed. Standing rule recorded in CLAUDE.md: core changes this repo
   needs are specs in this report for the typesymbolic agent — never edits.
2. **Question primitives (`jev.py`).** Local `Noul`/`Score` deleted (exact
   matches with core's, kept as re-exports since this module is the repo's
   judge boundary); `Choice` is a one-field subclass widening core's
   `criteria` to `dict[str, str | None]` — the real wire accepts null
   criteria ("None for undescribed labels", per `typesafe_sdk.Choice`), and
   this domain enumerates live candidates with no per-option text. An
   extension, not a shim: nothing converts at a boundary, this IS the type
   every call site uses; wire bytes pinned in `tests/test_wire_contract.py`.
   The widening is spec'd for upstream (see "Recommended core changes").
3. **Dead field deleted.** The old local `Noul.criteria` (a generic map) was
   never sent by any call site and is off-spec vs the API's real noul
   criteria (`{true, false}` outcome descriptions, per
   `typesafe_sdk.NoulCriteria`). Dead code is not migrated into a
   dependency; it dies.
4. **Gate (`judge/gate.py`).** Core's `GateVerdict`/`GateResult` are THE gate
   types (local copies deleted); every threshold comparison goes through
   core's one shared primitive, `circuit.threshold_decision`. Domain-owned
   and unchanged: argv read-only/deny classification, calibrated reason
   strings (journal continuity), ask orchestration. Verdict values changed
   approved/denied -> act/deny — grep-verified internal-only (journal
   outcome rows carry the MCP layer's own "approve"/"deny" literals,
   untouched). `noul_confidence` renamed to core's `confidence` across 9
   files. `finalize_gate` fails CLOSED on missing confidence — deliberately
   stricter than core's `mutation_gate`, which fails open on None (H3);
   documented at the call site and pinned by test.
5. **Vocabulary protocol conformance.** `QuestionSet` now satisfies
   `typesymbolic.vocab.Vocabulary` (`version` + `ask(qid, **slots)`), with
   criteria arriving via a `criteria=` slot (the protocol has no criteria
   channel; core's own `resolve_one` overwrites Choice criteria from
   `propose()` anyway). The richer native API (`text()`, fit-question
   families, per-call-site `choice()`) stays: the protocol is a subset
   surface a domain conforms to, not a replacement for its question
   discipline. Conformance makes this repo a real plugin for the
   vocabulary half without touching core.
6. **Pinning tests.** Wire-contract shapes byte-pinned; protocol conformance
   + fail-closed behavior pinned; core-gate-type identity + raw-noul
   confidence + fail-closed-on-None pinned.

**What was deliberately NOT done, and the why for each:**

- **Journal.** Core's committed journal lacks the capture surface, blob
  offloading, provenance channel, and a synchronous-write mode (all spec'd
  above). Extending here would mean a `DecisionJournal(Journal)` subclass
  overriding `_append` (core's is async-queue; this domain needs sync),
  `_write_row` (no offloading in core — the blob machinery would live here
  anyway), `record_*` (no capture fields), `_path` (day files) — almost
  none of core's journal code would execute. Inheritance theater: same
  domain code as today plus indirection, nothing deleted. Parked until the
  typesymbolic agent lands the capture-surface specs.
- **Engine boundary (`JudgeEngine`/`ask_all`/`resolve_one`).** Conformance
  would require an answer-conversion adapter (rejected earlier as a
  bridge/shim), and this domain's decision shapes — two-round narrowing,
  fused pick+gate, gate-only — don't fit `resolve_one()`'s single-Choice
  shape anyway. No payoff today.
- **Wire answer models.** Switching to core's normalized `Answer` would
  change the journaled answer shape (a journal-era decision, see H1) and
  routes noul answers through the synthesized confidence (the H2 trap).
- **Transport.** Chained to the answer-shape decision; also the wire body
  is contract-frozen and core's `JevEngine` currently drops noul criteria
  entirely (spec'd above).
- **Calibration.** Journal-coupled (continuous.py reads raw rows) and keyed
  differently: per-engine profile knobs here vs per-(qid, engine,
  model_revision) store in core — a granularity decision, not a refactor.
- **`matching.py`, `budget.py`, `shadow.py`, `ledger.py`, narrowing.**
  Genuinely domain-owned (candidate narrowing heuristics, engine budget
  profiles, A/B observation, usage accounting); no core counterpart is
  expected or wanted.

**Results:** jevdevice 199/199 including the live on-device test; typesymbolic
untouched, 153 green; ruff clean both. Net ~50 duplicated lines deleted.

## What was migrated (attempted, live-verified, then reverted)

### 1. The typed question primitives (`Noul`/`Choice`/`Score`)

jevdevice's own pydantic question models are **deleted**;
`jevdevice.jev` now imports them from `typesymbolic.question` and re-exports
them (it remains the repo's judge-boundary import point). One core change was
required and made (see below): `Choice.criteria` loosened from
`dict[str, str]` to `dict[str, str | None]`.

Why this was clean: the models are pure data with one consumer (wire
serialization), and the serialized bytes are pinned by a new test
(`tests/test_wire_contract.py`) so a core change that alters the wire body
fails fast. **Lesson: pin the serialized shape of shared types in a test the
domain owns.** The "FROZEN wire contract" discipline survives a dependency
swap only if something executable enforces it.

A side effect worth noting: jevdevice's local `Noul` had a `criteria:
dict[str, str] | None` field that was *dead* — no call site ever sent it, and
it was off-spec anyway (the API's noul criteria is `{true, false}` outcome
descriptions, per `typesafe_sdk.NoulCriteria` — a TypedDict with
`true`/`false` keys, not a generic map). Migration deleted it rather than
carrying it into core. **Lesson: a domain's unused-but-declared fields are
spec noise; don't generalize them into the framework.**

### 2. The gate (`GateVerdict`/`GateResult`/threshold mechanics)

jevdevice's `judge/gate.py` no longer defines its own verdict class, result
dataclass, or threshold comparison. Core's `typesymbolic.gate.GateVerdict`/
`GateResult` are the only gate types, and every threshold comparison goes
through core's one shared primitive, `circuit.threshold_decision()`. What
remains domain-owned in `gate.py` is exactly what core should not know:
the argv read-only/deny classification, the calibrated reason strings
(`read_only`, `deny_listed`, `jev_confirmed`, `jev_uncertain`), and the
ask-orchestration (`gate_command`, `propose_from_closed_set`).

Why this worked: core's `GateResult` takes an arbitrary `reason` string, so
the domain could keep its journal-continuity vocabulary while adopting core's
mechanics and verdict values (`act`/`needs_approval`/`deny` replaced
`approved`/`needs_approval`/`denied` everywhere — internal-only; verified by
grep that no journal column or eval tooling keys off the old strings; the
MCP surface's own literals `approve`/`deny` in outcome rows are untouched).
**Lesson (positive): separating gate *mechanics* (core) from gate
*vocabulary* (domain) is the right factoring. Make the vocabulary
overridable, not just the threshold.**

## What was hard, and why

### H1. The journal — the central blocker, and a design tension, not a bug

This is the one that stops the migration from going deeper, and it deserves
the most attention because it is a *framework* lesson, not a jevdevice
quirk.

**jevdevice's journal and typesymbolic's journal are two different products
that happen to share a name.**

- typesymbolic's journal is a **calibration label index**: minimal rows
  (ids, answers, verdicts — "never a domain payload"), a live per-`(qid,
  engine, model_revision)` index of `(confidence, verified)` pairs, built to
  drive inline recalibration cheaply.
- jevdevice's journal is a **replayable decision-capture artifact**: every
  decision row carries the full `state`, the full `questions` wire dict, the
  full answer *distribution* (probabilities/confidence/noul, not just the
  winning pick), plus per-ask provenance — `phase`, `goal_id`/`goal` (via a
  contextvar scope), `call_id`, `truncation`, `generated` (escape-hatch
  question provenance), `shadow_of` (A/B observation rows), `usage` deltas,
  `elapsed_ms`. Big values (screen dumps, raw probe text) go to a
  content-addressed blob store; rows keep refs. Its documented invariants are
  REPLAY (rows reconstruct exactly what `ask()` sent and got) and
  BIG-VALUES-OUT-OF-LINE.

The rich shape is not optional decoration — it is the raw material for
roughly 4,300 lines of eval tooling: fine-tune export, the flywheel
(promoting journaled runtime-generated questions into the next compiled
question set), recipe building (foreground-app graph edges), question-set
compilation (witness counts stamped back into the frozen YAML), the shadow
A/B harness, and the audits. Adopting core's journal as-is would break all
of that; adopting it with a second domain journal beside it halves the value
of the unification.

`record_snapshot()` is not an answer here in its current form: it is opaque
and unindexed, while the consumers above need to *join and filter* by
`call_id`, `qid`, `phase`, `goal_id`, `shadow_of`, `generated`.

**Recommendation for core:** treat rich decision capture as a first-class
concern with a deliberately narrow contract, e.g. an optional, indexed
`extras`/`capture` field on decision rows with out-of-line blob offloading
above a size floor (jevdevice's `BlobStore` is ~30 lines and transfers
directly), plus a documented channel for per-ask metadata. The live
calibration index should keep keying off only what it needs — nothing about a
richer row forces the index to grow. The alternative framing, "domains keep
their own capture journal and core keeps the label journal", is workable but
then `ask_batch()`'s built-in journaling (opt-in, minimal) stops being the
shared path and every domain reimplements the write — the drift the package
exists to prevent.

### H2. Never synthesize a confidence for a question type that doesn't carry one

Core's normalized `Answer` populates `confidence` for a noul as
`|p − 0.5| × 2`. jevdevice's gates threshold on the **raw noul probability**
(the 0.8 gate threshold was calibrated as "p ≥ 0.8 auto-approves"). These are
different scales: the same 0.8 threshold against the synthesized value means
"p ≥ 0.9 *or* p ≤ 0.1". A domain that swaps its gate inputs from raw noul to
core `Answer.confidence` **silently reinterprets every calibrated threshold** —
no error, no test failure, just a different gate.

jevdevice avoided the trap by keeping its wire answer models (see H4), but
any future `resolve_one()`-style consumer gating on a noul via core's Answer
hits it. **Recommendation for core:** either leave `confidence` None for
nouls (and make gates handle None explicitly — see H3), or keep the raw
probability as the field gates are told to read, and *document the synthesis
as non-calibrated*. A framework whose pitch is "gates mean what they say
against a calibrated confidence" cannot quietly define that confidence
differently per question type.

### H3. `mutation_gate` fails OPEN on missing confidence

`mutation_gate(None, threshold)` returns verdict `ACT` ("no confidence to
gate"). For a *mutation* gate, missing evidence should defer to the human
(`needs_approval`), not approve. jevdevice's local `finalize_gate` now fails
closed on None (`passes is True` required) — deliberately *more* conservative
than core's named gate. **Recommendation for core:** flip the None branch in
`mutation_gate` to needs-approval (keep `claim_gate`'s semantics whatever
repo-activity needs — verify with that domain), or at minimum make the
fail-open behavior loud in the docstring.

### H4. The transport/answer/journal cluster — one decision, not three migrations

The next obvious migration is deleting jevdevice's hand-rolled HTTP client
(~120 lines: retry on 429/529, wire parsing) in favor of core's `JevEngine`
over `typesafe-sdk`. It is chained to H1/H2:

`JevEngine.ask_all()` returns core's normalized `Answer`s → jevdevice journals
`answer.model_dump()` of what it got → **the journaled answer shape changes**
(qid added, synthesized confidence added for nouls, `abstained` flag added)
→ REPLAY byte-comparability across the migration boundary breaks, and eval
parsers that read old and new rows together see a field whose meaning
differs by question type (H2). Converting core Answers back to wire dicts
for journaling would be a de-normalizing shim — rejected.

So: transport migration is *possible* but must be taken together with a
journal-era decision (H1). It also needs one live verification (the repo's
single live test needs a reachable phone), since the wire body is contract-
frozen. Related finding while checking feasibility: **core's `JevEngine`
currently drops noul criteria entirely** (`_to_sdk_question` builds
`sdk.Noul(instructions=...)` and nothing else). Harmless for jevdevice (it
never sends noul criteria), but it means core's question model and core's
transport disagree about the wire: if `Noul` ever grows the API's
`{true, false}` criteria, `_to_sdk_question` must forward it.

### H5. The Vocabulary protocol is too narrow for live-enumerated candidates

jevdevice's `question_sets` is its frozen vocabulary — YAML artifacts,
fail-closed slot validation, compiled from the journal with witness counts.
Core's `Vocabulary.ask(qid, **slots)` cannot express what it needs:

- **criteria at ask time**: a Choice's candidates are live-enumerated runtime
  data; the wording is frozen, the option set is not. `FrozenVocabulary`
  stores criteria *in the entry* — the opposite assumption. `resolve_one()`
  papers over this by overwriting Choice criteria from `propose()`, but every
  direct ask (narrowing rounds, closed-set picks, fused pick+gate) supplies
  candidates at the call site.
- **wording extraction** (`text(qid, **slots)`): jevdevice threads formatted
  wording into shared builders (the fit-question family: one "does this
  candidate fit" Noul *per candidate*, `{candidate}` filled per-instance).
  The protocol only returns Questions, not the text in between.
- **escape-hatch provenance**: runtime-generated questions (the promotion
  path into the next compiled set) carry `generated=<source>` on the decision
  row. No Vocabulary-level concept for this.

For now `question_sets` remains jevdevice's vocabulary implementation and
simply doesn't go through the core protocol — which is fine, but it means the
"implement `Vocabulary`, get everything for free" story doesn't hold for the
domain with the most developed question discipline. **Recommendation for
core:** add criteria-supply to the ask surface (e.g. an optional
`criteria=`/`candidates=` parameter with a documented default from the
entry), and consider whether `text()` belongs in the protocol. Fit-question
families (one question per candidate, from one frozen wording) look like a
genuinely general pattern worth owning in core — both round-2 narrowing and
the flywheel's promotion path depend on them.

### H6. Calibration keying: per-engine profiles vs per-(qid, engine, revision) store

jevdevice's thresholds live in `budget.py` as per-engine profile knobs
(`gate_threshold`, `min_confidence`, `min_margin`, `min_fit`, `noul_floor`),
fit offline by eval harnesses with MIN_PRECISION 0.95 / MIN_LABELS 20. Core's
`CalibrationStore` keys per `(qid, engine, model_revision)` with inline
tighten-only recalibration (MIN_PRECISION 0.9 default). Both are defensible —
but they are different models of "what is calibrated against what": jevdevice
calibrates a *scale* shared by every question of a type on an engine; core
calibrates *per question*. Keying jevdevice's gate threshold per-qid would
fragment the label count n across qids that share one scale. Migrating
calibration is therefore not mechanical; it needs a decision about which
granularity is real, and it is also journal-coupled (H1: continuous.py reads
raw replay rows). Deferred, documented.

### H7. Small but real friction

- **uv path dependencies snapshot.** Editing typesymbolic does not propagate
  into a dependent's venv until `uv sync --reinstall-package typesymbolic`.
  During cross-repo iteration this looks like "my core change broke
  nothing and fixed nothing" — confusing. Anyone iterating across both repos
  should know this; longer-term, a uv workspace would fix it.
- **Re-export idioms are lint-hostile.** `from x import Y as Y` (the PEP 484
  re-export form) trips PLC0414; the workable form is a plain import plus
  `__all__`. Fine, just worth knowing when a module wants to be a boundary.
- **Core changes were needed even for the cleanest slice.** The two
  subsystems that migrated first (question types, gate) still required a core
  change (`Choice.criteria` nullable). Expect this pattern: the domain finds
  the wire truth, core generalizes it.

## Core changes implemented during the attempt — reverted; treat as tested specs

These were implemented locally, passed both repos' full suites plus the live
on-device pipeline, and were reverted per the operator decision above. Each
is a concrete, exercised specification for the typesymbolic agent to own:

1. `question.py`: `Choice.criteria` loosened to `dict[str, str | None]` —
   matches the real API (`typesafe_sdk.Choice`: "`None` for undescribed
   labels"). All 153 core tests passed unchanged.
2. **The journal capture surface** (addresses H1): `record_decision` grew
   optional `state` / `questions` / `extra` capture fields and
   `record_outcome` an `extra` dict merged flat into the row — journaled
   verbatim, returned by `replay()`, **never read by the calibration
   index** (`_index_row` touches only label fields). Big values go
   out-of-line via a `BlobStore` (content-addressed sha256, layout
   identical to the device domain's historical store so old blob refs
   resolve unchanged): `offload()` walks children-first and swaps anything
   over `blob_min_bytes` (2048 default, 0 disables) for
   `{blob: {sha256, bytes, encoding}}` refs; `replay()` resolves them
   back. Writer policy knobs: injectable `clock`, `fsync` flag,
   `background_writes` (default True = async queue writer; False = rows
   durable at `record_*` time — a provenance journal and a clock-injected
   test both need the synchronous mode). With these, the device domain's
   journal became a core-Journal subclass owning only its row schema,
   day-file policy, pruning and env knobs — its writer, blob machinery and
   resolver code deleted, with the full 2,528-row journal history
   replaying unchanged.
3. **A bug any such implementation must avoid, found by live probing:** a
   writer that offloads the row *as a whole* atomizes live decision rows —
   many MEDIUM fields exceed the threshold in total — into bare
   `{"blob": ...}` lines with no type/ts/call_id skeleton, silently
   skipped by the index rebuild. Both offline suites missed it (fixtures
   had a single big field; only real traffic has many medium ones). Offload
   **fields, never the row**; `_rebuild_index` must be defensive (torn or
   foreign lines contribute nothing; bare-ref lines recovered by resolving
   them); `_write_row` must be fail-open across the whole row including
   the offload — an I/O failure inside it otherwise kills the writer
   thread and hangs every future `flush()`. Also noted: a single-candidate
   Choice literally named `blob` with a dict description containing
   `sha256` would alias the ref marker — unreachable where descriptions
   are `str | None`, but a reserved marker key would close it.

## Recommended core changes, in priority order

1. **Journal capture surface (H1)** — **steps 1+2 landed** (capture fields,
   BlobStore, and the writer itself: policy knobs for clock/fsync/
   background-writes; the device domain's journal is a core-Journal
   subclass owning only schema + file policy). Remaining, and these are
   the genuinely hard parts: the ~4,300 lines of eval tooling coupled to
   the row schema, the journal-history era decision, and the label
   semantics (the device domain's `verification` field and noul answers
   vs core's `verify_status` + confidence-only index — ties into H2's
   no-synthesized-confidence fix).
2. **No synthesized noul confidence (H2)** + **fail-closed `mutation_gate`
   on None (H3)** — both are safety-semantics fixes, small diffs.
3. **Vocabulary: criteria-at-ask (H5)**, and consider owning the
   fit-question-family pattern.
4. **`JevEngine`: forward noul criteria** if/when `Noul` grows it (H4).
5. Consider a uv workspace for the sibling repos (H7).

## Elimination roadmap: what is gated on what

The remaining migration surface, and the specific gate for each. Recorded
here because the gates are NOT all one thing, and two of the three are this
repo's decisions, not core's:

| Elimination target | ~lines | Gate |
|---|---|---|
| Journal writer + blob machinery | 270 | **G1: core capture surface** (spec above) |
| Transport (`jev.py` HTTP client) | 200 | **G2: the answer-shape era decision** (ours — core can ship nothing that unblocks it) |
| Wire answer models | 40 | G2 (same decision) |
| Calibration overlap (`continuous.py` etc.) | 300 | **G3: label semantics** (core's synthesized noul confidence, H2) + keying granularity (a joint design decision) |
| Vocabulary fit-families into core | small | Nothing — but it moves code INTO core more than out of here |

G1 is a missing core feature (its agent's call). G2 is a journal-era
comparability decision this repo must make deliberately (old rows vs new
rows in A/B and audits) regardless of what core ships. G3 is a core
correctness issue plus a per-engine-scale vs per-(qid, engine, revision)
granularity agreement.

**The one ungated work item, and it is ours:** ~4,300 lines of eval tooling
read the journal's raw row schema directly (one reached into a private
method — since fixed to the public resolver, but the schema coupling
remains). That coupling is the cost multiplier on every schema move above,
core-driven or not. A thin journal query API in jevdevice (consumers stop
touching rows; readers go through one module) is pure domain-side work,
needs zero core changes, and shrinks every future migration step. It is the
recommended next piece of work in this repo, in-bounds by the standing rule.

## State and next steps

**Round 1 re-landed (operator-approved): jevdevice uses typesymbolic as a
dependency — question primitives (one justified subclass), gate types and
threshold mechanics, Vocabulary protocol conformance — with ZERO changes to
typesymbolic.** The journal, engine boundary, answers, transport, and
calibration stay domain-owned and parked, each with its reason recorded
above. The core changes that would unblock those remain SPECS in this report
for the typesymbolic agent: the capture surface (with the row-skeleton
offload bug any implementation must avoid), the noul-confidence fix, the
fail-closed mutation gate, and the Choice.criteria widening.

Surviving artifacts from the earlier attempt: this report, and two
malformed-but-replayable whole-row blob-ref lines in
`~/.jevdevice/journal/journal-20260922.jsonl` (written by the interim writer
during authorized live test runs; the restored code replays them fine;
cleanup left to the operator per the append-only invariant).
