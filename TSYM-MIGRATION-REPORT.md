# typesymbolic integration — jevdevice

jevdevice runs on `typesymbolic` as a local path dependency, pinned to a
committed revision (see the `[tool.uv.sources]` comment in `pyproject.toml`).
**jevdevice never edits typesymbolic.** A core change this repo needs is
written here as a spec for typesymbolic's own agent, never applied directly.

## What jevdevice uses from typesymbolic

- **Question primitives.** `jev.py` re-exports `Noul`, `Choice`, `Score`,
  `Question`, `QuestionRef`, and `Answer` unchanged; it is the repo's one
  judge-boundary import point. `Choice.criteria` is `dict[str, str | None]`
  in core itself, so a live-enumerated candidate with no description needs
  no local subclass.
- **Engine + transport.** `common.py` builds a `typesymbolic.judge.JevEngine`
  (the HTTP client, retries, and wire parsing all live in core);
  `jev.py`'s `ask()` calls `typesymbolic.engine.ask_batch()` to run one
  batched ask, journal it, and feed the calling engine's usage ledger.
- **Journal.** `journal.decision_log.get_journal()` returns a
  `typesymbolic.journal.Journal` instance (default dir
  `~/.jevdevice/tsjournal/`, `JEV_JOURNAL_DIR` overrides). Row capture,
  replay, and blob offload for large values are core's; `decision_log.py`
  adds only goal-scope context (a contextvar read by `ask()`), jevdevice's
  env knobs, and retention pruning, which core does not do.
- **Outcome types.** `journal.outcomes.record_action()` builds
  `typesymbolic.domain.ActOutcome` / `ActStep` / `Verdict` rows from a
  device response and files them through the same core `Journal`.
- **Gate mechanics.** `GateVerdict` / `GateResult` (`typesymbolic.gate`) are
  the only gate types; every threshold comparison goes through
  `circuit.threshold_decision()`. `judge/gate.py` keeps only what core
  should not know: argv read-only/deny classification and the calibrated
  reason vocabulary (`read_only`, `deny_listed`, `jev_confirmed`,
  `jev_uncertain`).
- **Vocabulary protocol.** `question_sets` conforms to
  `typesymbolic.vocab.Vocabulary` (`version` + `ask(qid, **slots)`), so it
  is a real plugin for the vocabulary half of core.
- **Calibration.** `calibrate/units.py` reads a threshold through
  `typesymbolic.calibrate.current_threshold()` (`pool_revisions=True`)
  against a `typesymbolic.calibration_store.CalibrationStore`;
  `calibrate/continuous.py` refits through `typesymbolic.calibrate.recalibrate()`,
  tighten-only unless `JEV_PROMOTION_SIGNOFF` is set.
- **Planner episodes.** `execution/planner.py` opens one
  `typesymbolic.engine.Episode` per `resolve()` call.

## What stays domain-owned

No core counterpart exists or is wanted for: `matching.py` (candidate
narrowing heuristics), `budget.py` (per-engine budget/threshold profiles),
`ledger.py` (usage accounting), `judge/narrowing.py` (two-round candidate
narrowing), `judge/shadow.py` (A/B shadow-engine plumbing), and
`question_sets`' richer query surface (`text()`, fit-question families,
escape-hatch provenance) — the Vocabulary protocol only covers `ask()`.

## Open items

None verified. Every gap an earlier round of this report raised against
core is closed at the pinned commit:

- **Noul confidence is never synthesized.** `question.py`'s `Answer`
  docstring: "`confidence` is the judge's own reported value for
  choice/score; `None` for a noul -- never synthesized" (line 69); `value()`
  reads the raw `noul` field for `scale="noul_p"` (lines 87-88). `gate.py`
  reading the raw noul probability is simply reading the only value core
  ever puts there, not working around a divergence.
- **`mutation_gate` fails closed on missing confidence.** `gate.py`'s
  `_confidence_verdict` (line 59): `if passes is None: return
  GateResult(settled_uncertain, "no confidence to gate", ...)`, and
  `mutation_gate` passes `uncertain_verdict=GateVerdict.NEEDS_APPROVAL`
  (line 78), so a missing confidence already yields `needs_approval`,
  matching jevdevice's own `finalize_gate`.
- **Vocabulary takes criteria at ask time.** `vocab.py`'s `Vocabulary`
  protocol: `def ask(self, qid: str, *, criteria: Mapping[str, str | None]
  | None = None, **slots: str) -> Question` (line 25); `FrozenVocabulary.ask()`
  applies a given `criteria` over the entry's own (lines 131, 144-148).
- **The journal capture surface landed.** `engine.py`'s `ask_batch()` takes
  `capture`, and journals `state`/`asked`/`extra`/`scope` on every decision
  row (lines 41-44, 50-51, 60, 71).
- **Calibration pools by domain-chosen unit, not forced per-qid.**
  `calibration_store.py`'s `CalibrationStore.key()` is `f"{name}|{engine}|
  {model_revision or ''}"` (line 30), where `name` is `calibrate/units.py`'s
  own `(group, scale)` pair (e.g. `gate_threshold` pools onto `"gate"`) --
  jevdevice already gets the per-engine-scale pooling it wants.

Evidence is against typesymbolic commit `83d312b` (the pin in
`pyproject.toml`'s `[tool.uv.sources]` comment). Re-verify against the code
at that path before citing any of this, not from memory.
