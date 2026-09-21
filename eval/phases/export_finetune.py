"""Export verified dev-half journal rows -> Laya fine-tune examples.

The decision journal is the labeled-example source. This exporter joins primary decision
rows to their ground truth and emits one JSONL example per journal row:

    {"call_id", "goal_id", "goal", "phase", "engine", "journal_file", "ts",
     "state", "questions" (frozen wire dicts), "supervision", "provenance"}

Supervision labels (one of):
  {"type": "noul",  "label": true|false}          -- device/hand ground truth
  {"type": "choice", "correct_option": "..."}     -- the verified/hand-labeled option
  {"type": "score", "level": <int>}               -- verified outcome level (no rows yet)

Ground-truth sources (all dev-half derived; the heldout half of eval/goals.yaml
is asserted absent from every exported row):
  1. dev_tap        -- the calibrate-CLI case tables (mirrored by
                       eval/phases/recalibrate_thresholds.py; hand-labeled dev taps).
  2. live_verified  -- decision rows joined by call_id to a device-verified
                       outcome row whose executed_command == proposed_command.
  3. narrow_cases   -- round-1 sweep rows for goals whose correct candidate is
                       in the same hand-labeled case tables.

EXCLUDED (counted in the card): shadow rows (shadow_of set), errored rows,
escalated/human-resolved rows, kind picks (their only labels are self-reported
-- never gate on self-reported confidence), the fused tap pick's safe_/fit_
questions (their chosen_action/proposed_command/target_bounds slots are not
reconstructable from the journal row), laya A/B-arm rows (escalations:
negative evidence only), and score questions (no rows exist yet).

The Kaggle fine-tune script explodes each exported row into one training
sequence per supervised question -- laya's build_sequence encodes ONE question
per sequence, so supervision granularity is (state, question, label).

Run: uv run python eval/phases/export_finetune.py [journal_dir]
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# The label tables + matching logic come from the refit harness (single source
# of truth, reused -- a drift-guard test pins the mirror copy).
import recalibrate_thresholds as p4r

from jevdevice.budget import NONE_OF_THESE
from jevdevice.decision_log import DecisionJournal

OUT_DIR = REPO / "eval" / "phases" / "finetune"

# named knobs
VAL_FRACTION = 0.1                  # goal-level hold-back for the fine-tune's own
                                    # val split (NOT the sacred goals.yaml heldout)
SPLIT_SALT = "phase6-finetune-v1"   # any change reshuffles the dev/val split; recorded in the card
MAX_EXAMPLES_PER_PHASE: int | None = None  # balance-phase cap; None = no cap at current n
SPOT_CHECK_N = 10                   # examples re-derived from the journal (acceptance spot-check)
SPOT_CHECK_SEED = SPLIT_SALT


def dev_heldout_goals() -> tuple[set[str], set[str]]:
    """(dev, heldout) goal texts from the frozen eval/goals.yaml split."""
    import yaml

    goals = yaml.safe_load((REPO / "eval" / "goals.yaml").read_text())
    dev, heldout = set(), set()
    for section in ("pc_questions", "phone_questions", "phone_actions"):
        for entry in goals.get(section, []):
            (dev if entry["split"] == "dev" else heldout).add(entry["goal"])
    return dev, heldout


def _criterion_matches(candidate: str, truth: tuple[str, ...]) -> bool:
    return any(t in candidate.casefold() for t in truth)


def _state_candidates(state: dict) -> list[str]:
    """Candidates in ask order, as the fit questions named them (laya profile:
    criteria VALUES are the rich descriptions the question embedded)."""
    cands = state.get("candidates")
    if isinstance(cands, dict):
        return [(v if isinstance(v, str) and v else k) for k, v in cands.items()]
    if isinstance(cands, list):
        return [str(c) for c in cands]
    return []


def _chunk_truth(goal: str, candidates: list[str]) -> tuple[str | None, bool]:
    """(correct option present in this chunk | None, does_any_fit) for round 1.
    Truth absent, or absent from this chunk, -> none_of_these / any=False."""
    truth = p4r.NARROW_PICK_CASES.get(goal)
    if truth is None:
        return None, False
    hits = [c for c in candidates if _criterion_matches(c, truth)]
    return (hits[0] if hits else None), bool(hits)


def label_round1_row(row: dict) -> dict[str, dict]:
    """Round-1 sweep rows (_score_chunk): 'any' Noul + 'pick' Choice over the chunk."""
    state = row.get("state") or {}
    goal = state.get("goal")
    if goal not in p4r.NARROW_PICK_CASES:
        return {}
    candidates = _state_candidates(state)
    correct, any_fits = _chunk_truth(goal, candidates)
    answers = row.get("answers") or {}
    out: dict[str, dict] = {}
    if "any" in answers:
        out["any"] = {"type": "noul", "label": any_fits}
    pick = answers.get("pick") or {}
    options = list(pick.get("probabilities") or {})
    if pick and options:
        if correct is not None and correct in options:
            out["pick"] = {"type": "choice", "correct_option": correct}
        elif correct is None and NONE_OF_THESE in options:
            out["pick"] = {"type": "choice", "correct_option": NONE_OF_THESE}
        # else: this chunk can't express the right answer -- unjudgeable, skipped
    return out


def label_ground_row(row: dict) -> dict[str, dict]:
    """Round-2 rows (narrow_and_pick): fit_i Nouls per candidate + 'pick' Choice.
    fit_i is True exactly for the correct candidate; every entry False for a
    genuinely-absent goal; abstain is the correct pick exactly then."""
    state = row.get("state") or {}
    goal = state.get("goal")
    if goal not in p4r.NARROW_PICK_CASES:
        return {}
    truth = p4r.NARROW_PICK_CASES[goal]
    answers = row.get("answers") or {}
    candidates = _state_candidates(state)
    out: dict[str, dict] = {}
    for key, answer in answers.items():
        if not key.startswith("fit_") or not isinstance(answer, dict) or answer.get("type") != "noul":
            continue
        index = int(key.split("_")[1])
        if index >= len(candidates):
            continue
        out[key] = {"type": "noul", "label": bool(truth) and _criterion_matches(candidates[index], truth)}
    pick = answers.get("pick")
    if pick and pick.get("type") == "choice":
        options = list(pick.get("probabilities") or {})
        if truth is None:
            if NONE_OF_THESE in options:
                out["pick"] = {"type": "choice", "correct_option": NONE_OF_THESE}
        else:
            hits = [c for c in options if _criterion_matches(c.casefold(), truth)]
            if hits:  # option set must be able to hold the answer
                out["pick"] = {"type": "choice", "correct_option": hits[0]}
    return out


def tap_table_label(state: dict, command: str) -> bool | None:
    """Dev-tap gate table lookup: exact (command, target_bounds) or (action, command)."""
    if "input tap" in str(command):
        bounds = str(state.get("target_bounds") or "")
        return next((y for cmd, b, y in p4r.TAP_NOUL_CASES if cmd == command and b == bounds), None)
    return next((y for action, cmd, y in p4r.GATE_NOUL_CASES
                 if cmd == command and state.get("chosen_action") in (None, "", action)), None)


def label_gate_row(row: dict, outcome: dict | None) -> dict[str, dict]:
    """gate 'safe' Noul: dev-tap table first; a live row joins its device-verified
    outcome by call_id. A human-corrected recovery_command means the proposal did
    NOT run as proposed -- the verified execution then proves nothing about the
    judge's proposal, so the row is skipped."""
    safe = (row.get("answers") or {}).get("safe")
    if not safe or safe.get("type") != "noul":
        return {}
    state = row.get("state") or {}
    command = state.get("proposed_command") or state.get("command")
    if command is None:
        return {}
    label = tap_table_label(state, str(command))
    if label is None and outcome is not None:
        recovery = outcome.get("recovery_command")
        proposed_ran = recovery in (None, "", command)
        if outcome.get("verification") == "verified" and proposed_ran:
            label = True
    if label is None:
        return {}
    return {"safe": {"type": "noul", "label": bool(label)}}


def is_dev_tap_gate(goal: str | None) -> bool:
    return goal in {action for action, _, _ in p4r.GATE_NOUL_CASES}


def example_for(row: dict, journal_file: str, outcomes: dict[str, dict]) -> dict | None:
    """(provenance, supervision) for one journaled primary decision, or None."""
    phase = row.get("phase")
    supervision: dict[str, dict] = {}
    provenance: str | None = None
    if phase == "gate":
        supervision = label_gate_row(row, outcomes.get(row.get("call_id")))
        if supervision:
            provenance = "dev_tap" if is_dev_tap_gate(row.get("goal") or "") else "live_verified"
    elif phase == "ground":
        supervision = label_ground_row(row)
        provenance = "dev_tap" if supervision else None
    elif phase == "recall":
        supervision = label_round1_row(row)
        provenance = "narrow_cases" if supervision else None
    if not supervision:
        return None
    questions = row.get("questions") or {}
    supervision = {key: label for key, label in supervision.items() if key in questions}
    if not supervision:
        return None
    return {
        "call_id": row["call_id"],
        "goal_id": row.get("goal_id"),
        "goal": row.get("goal") or (row.get("state") or {}).get("goal"),
        "phase": phase,
        "engine": row.get("engine"),
        "journal_file": journal_file,
        "ts": row.get("ts"),
        "state": row.get("state"),
        "questions": questions,
        "supervision": supervision,
        "provenance": provenance,
    }


def dedupe_key(example: dict) -> str:
    return hashlib.sha256(json.dumps(
        {"s": example["state"], "q": example["questions"], "sup": example["supervision"]},
        sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()


_HELDOUT: set[str] | None = None


def _heldout_casefold() -> set[str]:
    global _HELDOUT
    if _HELDOUT is None:
        _, heldout = dev_heldout_goals()
        _HELDOUT = {g.casefold() for g in heldout}
    return _HELDOUT


def is_val(goal_id: str | None, goal: str | None) -> bool:
    material = f"{SPLIT_SALT}:{goal_id or goal}"
    return int(hashlib.sha256(material.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < VAL_FRACTION


def count_distribution(examples: list[dict]) -> dict[str, int]:
    counts: Counter = Counter()
    for example in examples:
        for qname, label in example["supervision"].items():
            kind = label["type"]
            polarity = ""
            if kind == "noul":
                polarity = "true" if label["label"] else "false"
            elif kind == "choice":
                polarity = "abstain" if label["correct_option"] == NONE_OF_THESE else "pick"
            counts[f"{example['phase']}.{qname.split('_')[0]}.{kind}.{polarity}".removesuffix(".")] += 1
    return dict(sorted(counts.items()))


def spot_check(journal: DecisionJournal, examples: list[dict]) -> int:
    """Re-derive SPOT_CHECK_N exported examples straight from their journal rows."""
    random.seed(SPOT_CHECK_SEED)
    sample = random.sample(examples, min(SPOT_CHECK_N, len(examples)))
    wanted_files = {e["journal_file"] for e in sample}
    rows: dict[tuple[str, str], dict] = {}
    for path in sorted(journal.directory.glob("journal-*.jsonl")):
        if path.name not in wanted_files:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "decision" and row.get("call_id"):
                rows[(path.name, row["call_id"])] = journal._resolve_value(row)
    ok = 0
    for example in sample:
        row = rows.get((example["journal_file"], example["call_id"]))
        assert row is not None, f"spot-check: row {example['call_id']} not found in {example['journal_file']}"
        assert row.get("state") == example["state"], f"spot-check state drift on {example['call_id']}"
        assert row.get("questions") == example["questions"], f"spot-check question drift on {example['call_id']}"
        rederived = example_for(row, example["journal_file"], {})
        assert rederived is not None, f"spot-check: {example['call_id']} no longer exports"
        assert rederived["supervision"] == example["supervision"], f"spot-check supervision drift on {example['call_id']}"
        ok += 1
    return ok


def collect(journal: DecisionJournal, exclude_goal_ids: frozenset[str] = frozenset()) -> tuple[list[dict], dict[str, dict], Counter]:
    """Primary, answered decision rows (+ outcome index), with exclusion counts."""
    outcomes: dict[str, dict] = {}
    rows: list[tuple[str, dict]] = []
    dropped: Counter = Counter()
    for path in sorted(journal.directory.glob("journal-*.jsonl")):
        day = path.stem.removeprefix("journal-")
        for raw in journal.replay(day=day):
            if raw.get("type") == "outcome" and raw.get("call_id"):
                outcomes[raw["call_id"]] = raw
            elif raw.get("type") == "decision":
                rows.append((path.name, raw))
    examples: list[dict] = []
    for name, row in rows:
        if row.get("shadow_of"):
            dropped["shadow_row"] += 1
            continue
        if row.get("error") or not row.get("answers"):
            dropped["error_or_no_answers"] += 1
            continue
        state = row.get("state") or {}
        goal = state.get("goal") if isinstance(state, dict) else None
        if goal and goal.casefold() in _heldout_casefold():
            # Fail-closed on contamination -- unless the operator explicitly
            # excluded this goal id (stray pre-guard rows, card-recorded), the
            # exact mechanism build_recipes.py uses. Never a silent default.
            if row.get("goal_id") not in exclude_goal_ids:
                raise SystemExit(f"HELDOUT CONTAMINATION: row {row['call_id']} carries held-out goal {goal!r}")
            dropped["operator_excluded"] += 1
            continue
        example = example_for(row, name, outcomes)
        if example is None:
            dropped["unjudgeable_or_excluded"] += 1
            continue
        examples.append(example)
    return examples, outcomes, dropped


def main() -> int:
    args = sys.argv[1:]
    # Operator-only exclusions (human sign-off, recorded in the card): a
    # held-out goal id whose rows pre-date this contamination guard (a stray
    # early run). Mirrors build_recipes.py's flag -- excluding is an operator
    # decision made by passing the flag here, never a silent default.
    exclude: set[str] = set()
    journal_dir: str | None = None
    i = 0
    while i < len(args):
        if args[i] == "--exclude" and i + 1 < len(args):
            exclude.add(args[i + 1])
            i += 2
        elif journal_dir is None and not args[i].startswith("--"):
            journal_dir = args[i]
            i += 1
        else:
            print(f"unknown argument {args[i]!r}", file=sys.stderr)
            return 2
    journal = DecisionJournal(journal_dir)
    examples, _outcomes, dropped = collect(journal, exclude_goal_ids=frozenset(exclude))

    # dedupe identical (state, questions, supervision): calibration reruns repeat cases
    seen: dict[str, dict] = {}
    for example in examples:
        key = dedupe_key(example)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen[key] = example
    examples = list(seen.values())

    # balance-phase cap knob (None = no cap at the current dataset size)
    if MAX_EXAMPLES_PER_PHASE is not None:
        budget: Counter = Counter()
        capped: list[dict] = []
        for example in examples:
            budget[example["phase"]] += 1
            if budget[example["phase"]] <= MAX_EXAMPLES_PER_PHASE:
                capped.append(example)
            else:
                dropped["phase_cap"] += 1
        examples = capped

    train = [e for e in examples if not is_val(e["goal_id"], e["goal"])]
    val = [e for e in examples if is_val(e["goal_id"], e["goal"])]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for example in rows:
                handle.write(json.dumps(example, separators=(",", ":"), default=str) + "\n")

    # spot-check BEFORE reporting success: re-derive from the journal
    checked = spot_check(journal, examples)

    provenance = Counter(e["provenance"] for e in examples)
    card = {
        "generator": "eval/phases/export_finetune.py",
        "journal_dir": str(journal.directory),
        "labeled_source": "decision journal, dev-half only (heldout asserted absent; shadow_of rows excluded)",
        "n_examples": len(examples),
        "n_train": len(train),
        "n_val": len(val),
        "n_questions_supervised": sum(len(e["supervision"]) for e in examples),
        "questions_all": count_distribution(examples),
        "questions_train": count_distribution(train),
        "questions_val": count_distribution(val),
        "provenance_counts": dict(sorted(provenance.items())),
        "goals": dict(sorted(Counter(str(e["goal"]) for e in examples).items())),
        "excluded_rows": dict(sorted(dropped.items())),
        "operator_exclusions": sorted(exclude),
        "knobs": {"VAL_FRACTION": VAL_FRACTION, "SPLIT_SALT": SPLIT_SALT,
                  "MAX_EXAMPLES_PER_PHASE": MAX_EXAMPLES_PER_PHASE,
                  "note": "val here is dev-half data held back from the fine-tune; "
                          "the sacred goals.yaml heldout half is never read at all"},
        "known_gaps": [
            "n=0 score examples: no score asks exist in the journal yet",
            ("n=0 verified kind examples: kind picks have no device-verified labels "
             "(escalated kinds are human-resolved; ungated kinds emit no outcome rows)"),
            "the fused tap row's safe_/fit_ slots are unjudgeable from the journal state",
            ("fit_noul positives are rare (25/220 in the labeled captures) -- class imbalance "
             "must be handled at fine-tune time (pos_weight knob in the Kaggle script)"),
            ("val split is empty at the current n: the goal-level split needs >= "
             f"{int(1 / VAL_FRACTION)} goals to hold one out; it fills as goals accumulate "
             "(the fine-tune script still saves a checkpoint -- the first eval's "
             "loss=0 becomes the best and its state is written)"),
        ],
        "spot_check": {"n": checked, "method": "re-derived from journal rows, state/questions/supervision byte-compared"},
    }
    (OUT_DIR / "dataset_card.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str))
    print(f"\nwrote {OUT_DIR}/train.jsonl ({len(train)} rows) and val.jsonl ({len(val)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())