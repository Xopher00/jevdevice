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

## Open items (core changes jevdevice still works around)

- **H2 — no synthesized noul confidence.** Core's normalized `Answer`
  derives noul confidence as `|p − 0.5| × 2`. jevdevice's gate thresholds
  are calibrated against the RAW noul probability instead, so `gate.py`
  reads it directly rather than through a synthesized `Answer.confidence`.
  Fix: core should leave `confidence` unset for nouls, or document the
  synthesis as non-calibrated.
- **H3 — `mutation_gate` fails open on missing confidence.** Core's
  `mutation_gate(None, threshold)` returns `ACT`. jevdevice's own
  `finalize_gate` fails CLOSED on a missing confidence instead
  (`needs_approval`), deliberately more conservative. Fix: flip the `None`
  branch in `mutation_gate` to `needs_approval`, or document the fail-open
  behavior loudly.
- **Vocabulary criteria-at-ask.** `typesymbolic.vocab.Vocabulary.ask()` has
  no channel for live-enumerated candidates; `question_sets` supplies
  criteria at the call site instead of through the protocol.
