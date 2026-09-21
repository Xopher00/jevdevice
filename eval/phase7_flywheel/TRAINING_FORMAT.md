# Flywheel training formats (version flywheel-v1)

Emitted by eval/phase7_export_flywheel.py from the decision journal
(dev-half only, shadow rows and held-out goals excluded).

## sgcd-v1 — loss on recovery actions only
One row per mined recovery pair (eval/phase7_mine_failures.py).
- `context.broken_prefix`: ordered spans BEFORE the failure (action/decision/
  chatter, each with its call_id). CONTEXT ONLY — excluded from loss.
- `context.broken_state`: the failure outcome row (status, reasons, attempted
  command, its call_id).
- `target`: the recovery side's spans up to the verified completion. LOSS IS
  COMPUTED HERE ONLY, weighted by role (see span weights below).
- `provenance`: "human_corrected" (device_approve recovery_command) or "retry".

## span-weighted-v1 — action/decision spans, not chatter
One row per device-verified trajectory.
- `spans`: ordered, each `{call_id, role, phase?, text, weight}`:
  - role=action   — an executed_command (or every scroll_to_find swipe)   weight 1.0
  - role=decision — a judge Choice pick (non-abstain) + its confidence    weight 1.0
  - role=chatter  — Noul evidence, reasons, verify satisfied text         weight 0.1
Rationale: uniform SFT over small models overfits the chatter; loss weighting
concentrates capacity on what actually moves the device.

Reproduce: `uv run python eval/phase7_export_flywheel.py` (journal-only).
