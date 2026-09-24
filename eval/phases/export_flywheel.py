"""Training-format exporters for the failure-recovery data.

Two exports, one version knob, one format doc (TRAINING_FORMAT.md + the dataset
card carry the same spec the logbook records):

  sgcd-v1            one row per recovery pair. The broken prefix is context
                     only (no loss on it); supervision is the recovery side's
                     action + decision spans. This is what makes a failure
                     useful: the model learns how to get out, not how to get
                     stuck.
  span-weighted-v1   one row per verified trajectory, flattened into spans
                     (action | decision | chatter) with per-role loss weights --
                     uniform weighting overfits the chatter.

Offline + journal-only; consumes mine_recoveries.py's recovery_pairs.jsonl
and the journal directly for trajectories. Dev-half only: held-out goal texts
are asserted absent from every emitted row. Shadow rows never enter.

  uv run python eval/phases/export_flywheel.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

from jevdevice.budget import NONE_OF_THESE
from jevdevice.journal import decision_log

OUT_DIR = REPO / "eval" / "phases" / "flywheel"
FORMAT_VERSION = "flywheel-v1"  # bump on any format change; the card + doc record it

# --- loss-weight knobs (named, never bare numbers at a call site) ---------------
WEIGHT_ACTION = 1.0    # a command that actually ran on the device
WEIGHT_DECISION = 1.0  # the judge's Choice pick (which app / element / service / kind)
WEIGHT_CHATTER = 0.1   # Noul evidence (fits, verifies), reasons, abstain text
import mine_recoveries as p7m

TRAINING_FORMAT_DOC = """\
# Flywheel training formats (version {version})

Emitted by eval/phases/export_flywheel.py from the decision journal
(dev-half only, shadow rows and held-out goals excluded).

## sgcd-v1 — loss on recovery actions only
One row per mined recovery pair (eval/phases/mine_recoveries.py).
- `context.broken_prefix`: ordered spans BEFORE the failure (action/decision/
  chatter, each with its call_id). CONTEXT ONLY — excluded from loss.
- `context.broken_state`: the failure outcome row (status, reasons, attempted
  command, its call_id).
- `target`: the recovery side's spans up to the verified completion. LOSS IS
  COMPUTED HERE ONLY, weighted by role (see span weights below).
- `provenance`: "human_corrected" (device_approve recovery_command) or "retry".

## span-weighted-v1 — action/decision spans, not chatter
One row per device-verified trajectory.
- `spans`: ordered, each `{{call_id, role, phase?, text, weight}}`:
  - role=action   — an executed_command (or every scroll_to_find swipe)   weight {w_action}
  - role=decision — a judge Choice pick (non-abstain) + its confidence    weight {w_decision}
  - role=chatter  — Noul evidence, reasons, verify satisfied text         weight {w_chatter}
Rationale: uniform SFT over small models overfits the chatter; loss weighting
concentrates capacity on what actually moves the device.

Reproduce: `uv run python eval/phases/export_flywheel.py` (journal-only).
"""


def _picked_answer(row: dict) -> dict | None:
    """The judge's Choice pick from one decision row, if any (abstains excluded
    -- an abstain is chatter, not a decision the model should copy)."""
    answers = row.get("answers") or {}
    for key, answer in answers.items():
        if isinstance(answer, dict) and answer.get("type") == "choice" and answer.get("choice") not in (None, NONE_OF_THESE):
            return {"question": key, "picked": answer["choice"], "confidence": answer.get("confidence")}
    return None


def _noul_chatter(row: dict) -> list[str]:
    answers = row.get("answers") or {}
    return [f"{key}={answer.get('noul')}" for key, answer in answers.items()
            if isinstance(answer, dict) and answer.get("type") == "noul"]


def spans_for(rows: list[dict]) -> list[dict]:
    """One trajectory's rows -> ordered weighted spans (span-weighted-v1)."""
    spans: list[dict] = []
    for row in rows:
        call_id = row.get("call_id")
        if row.get("type") == "decision":
            picked = _picked_answer(row)
            if picked:
                spans.append({"call_id": call_id, "role": "decision", "phase": row.get("phase"),
                              "text": json.dumps(picked, sort_keys=True), "weight": WEIGHT_DECISION})
            chatter = _noul_chatter(row)
            if chatter:
                spans.append({"call_id": call_id, "role": "chatter", "phase": row.get("phase"),
                              "text": ";".join(chatter), "weight": WEIGHT_CHATTER})
        else:  # outcome row: reasons live in the flat extra field act_reasons, not on decisions
            names = [s.get("name") for s in (row.get("steps") or []) if s.get("name")]
            for command in names or ([row["executed_command"]] if row.get("executed_command") else []):
                spans.append({"call_id": call_id, "role": "action", "text": command, "weight": WEIGHT_ACTION})
            if row.get("recovery_command"):
                spans.append({"call_id": call_id, "role": "action", "text": row["recovery_command"],
                              "weight": WEIGHT_ACTION})
            if row.get("act_reasons"):
                spans.append({"call_id": call_id, "role": "chatter",
                              "text": ";".join(str(r) for r in row["act_reasons"]), "weight": WEIGHT_CHATTER})
    return spans


def sgcd_row(pair: dict) -> dict:
    """One recovery pair -> sgcd-v1 row: broken prefix as context, loss on the
    recovery side's action+decision spans only."""
    journal_rows = list(decision_log.get_journal().replay())
    rows = p7m.trajectories_by_goal(journal_rows).get(pair["goal_id"], [])
    failure_ts = pair["broken"]["ts"]
    recovery_ts = pair["recovery"]["ts"]
    prefix_rows = [r for r in rows if (r.get("ts") or "") <= failure_ts]
    recovery_rows = [r for r in rows if failure_ts < (r.get("ts") or "") <= recovery_ts]
    recovery_spans = [s for s in spans_for(recovery_rows) if s["role"] in ("action", "decision")]
    return {
        "format": "sgcd-v1",
        "pair_goal_id": pair["goal_id"],
        "goal": pair["goal"],
        "provenance": pair["provenance"],
        "context": {
            "broken_state": pair["broken"],
            "broken_prefix": spans_for(prefix_rows),
        },
        "target": recovery_spans,
    }


def span_weighted_row(goal_id: str, rows: list[dict], goal: str, verdicts: dict[str, str]) -> dict | None:
    """One device-verified trajectory -> span-weighted-v1 row."""
    if not any(r.get("type") == "outcome" and verdicts.get(r.get("call_id")) == "verified" for r in rows):
        return None
    return {
        "format": "span-weighted-v1",
        "goal_id": goal_id,
        "goal": goal,
        "spans": spans_for(rows),
    }


def main() -> int:
    journal = decision_log.get_journal()
    journal_rows = list(journal.replay())
    verdicts = {r["call_id"]: r["status"] for r in journal_rows if r["type"] == "verdict"}
    heldout = p7m.heldout_goals()

    pairs_file = OUT_DIR / "recovery_pairs.jsonl"
    if pairs_file.exists():
        pairs = [json.loads(line) for line in pairs_file.read_text().splitlines() if line.strip()]
    else:
        pairs, _ = p7m.collect(journal)

    sgcd_rows = []
    for pair in pairs:
        goal = pair.get("goal") or ""
        assert goal.casefold() not in heldout, f"held-out goal reached the exporter: {goal!r}"
        sgcd_rows.append(sgcd_row(pair))

    weighted_rows = []
    for goal_id, rows in p7m.trajectories_by_goal(journal_rows).items():
        goal = next((r.get("goal") for r in rows if r.get("goal")), "")
        assert goal.casefold() not in heldout, f"held-out goal reached the exporter: {goal!r}"
        row = span_weighted_row(goal_id, rows, goal, verdicts)
        if row is not None:
            weighted_rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sgcd_train.jsonl").write_text(
        "".join(json.dumps(r, separators=(",", ":"), default=str) + "\n" for r in sgcd_rows))
    (OUT_DIR / "span_weighted_train.jsonl").write_text(
        "".join(json.dumps(r, separators=(",", ":"), default=str) + "\n" for r in weighted_rows))
    (OUT_DIR / "TRAINING_FORMAT.md").write_text(
        TRAINING_FORMAT_DOC.format(version=FORMAT_VERSION, w_action=WEIGHT_ACTION,
                                   w_decision=WEIGHT_DECISION, w_chatter=WEIGHT_CHATTER))

    span_counts: Counter = Counter()
    for row in weighted_rows:
        span_counts.update(span["role"] for span in row["spans"])
    card = {
        "generator": "eval/phases/export_flywheel.py",
        "format_version": FORMAT_VERSION,
        "loss_weights": {"action": WEIGHT_ACTION, "decision": WEIGHT_DECISION, "chatter": WEIGHT_CHATTER},
        "sgcd_rows": len(sgcd_rows),
        "span_weighted_rows": len(weighted_rows),
        "span_role_counts": dict(sorted(span_counts.items())),
        "guards": "dev-half only (heldout asserted absent); shadow_of rows excluded; "
                  "kind picks excluded as self-labeled; gate thresholds untouched",
        "format_doc": str(OUT_DIR / "TRAINING_FORMAT.md"),
    }
    (OUT_DIR / "export_card.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
