"""Score the fine-tuned judge against the base checkpoint on the val split.

Offline (zero device, zero live questions): replays the exported val examples
through each checkpoint's serving path (laya Agent.system_one -- the exact
encoding and configured temperature the runtime sees) and compares:

  - decision quality: argmax accuracy vs the verified label, per question type
    and per phase
  - probability quality: Brier / log-loss / ECE on the noul probability and on
    the choice probability vector (the P4 calibration recheck -- a fine-tune is
    a new confidence distribution, so temperatures/thresholds must be refit,
    never carried over)
  - temperature recheck: the p^(1/T) refit on choice vectors, old vs new

The scorecard (JSON) holds both checkpoints' numbers side by side.

Device: set JEV_DEVICE=cuda -- the single attached checkpoint fits the 4 GiB
card; CPU is minutes per batch and produces the same numbers.

Run: uv run python eval/phase6_eval_finetune.py \
    --ckpt <path-or-HF-revision-of-the-fine-tune> \
    [--base <path-or-HF-revision>] [--val eval/phase6_finetune/val.jsonl]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


_spec = importlib.util.spec_from_file_location("phase4_recalibrate", REPO / "eval" / "phase4_recalibrate.py")
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise SystemExit("cannot load the metric functions")
p4r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4r)

OUT_DIR = REPO / "eval" / "phase6_finetune_eval"


def resolve_checkpoint(spec: str, token: str | None) -> Path:
    """Local path, or a hub revision resolved offline-first like the runtime."""
    if Path(spec).exists():
        return Path(spec)
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download("convaiinnovations/laya", revision=spec,
                                      allow_patterns=["typed-decisions/*"], local_files_only=True, token=token))
    except OSError:  # not in the local HF cache: fetch once
        return Path(snapshot_download("convaiinnovations/laya", revision=spec,
                                      allow_patterns=["typed-decisions/*"], token=token))


def load_agent(location: Path, device: str):
    from laya.agent import Agent

    return Agent(str(location), subfolder="typed-decisions", device=device)


def score(agent, val_rows: list[dict], device: str) -> dict:
    """Run every val row through the serving path; returns per-type records."""
    noul_pairs: list[tuple[float, bool]] = []
    choice_rows: list[dict] = []
    by_phase: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in val_rows:
        questions = row["questions"]
        answers = agent.system_one(row["state"], questions)["answers"]
        for qname, supervision in row["supervision"].items():
            answer = answers[qname]
            phase = row.get("phase") or "unlabeled"
            if supervision["type"] == "noul":
                label = bool(supervision["label"])
                noul_pairs.append((float(answer["noul"]), label))
                by_phase[phase]["n"] += 1
                by_phase[phase]["correct"] += int((answer["noul"] >= 0.5) == label)
            elif supervision["type"] == "choice":
                correct = supervision["correct_option"]
                probabilities = answer["probabilities"]
                pick = answer["choice"]
                is_correct = pick == correct
                choice_rows.append({"phase": phase, "correct": is_correct, "correct_pick": correct,
                                    "probabilities": probabilities,
                                    "confidence": answer.get("confidence")})
                by_phase[phase]["n"] += 1
                by_phase[phase]["correct"] += int(is_correct)
            # score: no rows exist yet (dataset card); wired when they do
    return {"noul_pairs": noul_pairs, "choice_rows": choice_rows,
            "by_phase": {k: dict(v) for k, v in sorted(by_phase.items())}}


def noul_report(pairs: list[tuple[float, bool]]) -> dict:
    if not pairs:
        return {"n": 0}
    probabilities = [p for p, _ in pairs]
    labels = [y for _, y in pairs]
    error, _table = p4r.ece(probabilities, labels)
    return {
        "n": len(labels),
        "accuracy@0.5": sum(1 for p, y in zip(probabilities, labels) if (p >= 0.5) == y) / len(labels),
        "brier": p4r.brier(probabilities, labels),
        "logloss": p4r.logloss(probabilities, labels),
        "ece": error,
        "n_true": sum(labels),
    }


def choice_report(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    correct_flags = [r["correct"] for r in rows]
    confidences = [r["confidence"] for r in rows]
    fitted = p4r.fit_temperature(_vectors_with_names(rows))
    error, _table = p4r.ece(confidences, correct_flags)
    return {
        "n": len(rows),
        "pick_accuracy": sum(correct_flags) / len(rows),
        "confidence_brier": p4r.brier(confidences, correct_flags),
        "confidence_logloss": p4r.logloss(confidences, correct_flags),
        "confidence_ece": error,
        "temperature_refit": fitted,
    }


def _vectors_with_names(rows: list[dict]) -> list[tuple[dict, str]]:
    """(probabilities, correct option name) pairs; rows whose label is missing
    the option are skipped (the same rule the exporter applies)."""
    out = []
    for row in rows:
        pick_name = row.get("correct_pick")
        if pick_name is None:
            continue
        if pick_name in row["probabilities"]:
            out.append((row["probabilities"], pick_name))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="fine-tuned checkpoint: local dir or HF revision")
    parser.add_argument("--base", default=None, help="base checkpoint (default: the pinned runtime revision)")
    parser.add_argument("--val", type=Path, default=REPO / "eval" / "phase6_finetune" / "val.jsonl")
    parser.add_argument("--device", default=None, help="default: JEV_DEVICE, else cuda (CPU is minutes/batch)")
    parser.add_argument("--limit", type=int, default=None, help="cap val rows (smoke runs)")
    args = parser.parse_args()

    from jevdevice.laya_backend import REVISION

    device = args.device or os.environ.get("JEV_DEVICE", "cuda")
    rows = [json.loads(line) for line in args.val.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("no val rows -- nothing to score")
        return 1

    checkpoints = {"base": resolve_checkpoint(args.base or REVISION, None), "finetune": resolve_checkpoint(args.ckpt, None)}
    scorecard: dict = {"device": device, "n_val_rows": len(rows), "checkpoints": {}}
    for name, location in checkpoints.items():
        print(f"scoring {name}: {location}")
        agent = load_agent(location, device)
        report = score(agent, rows, device)
        scorecard["checkpoints"][name] = {
            "noul": noul_report(report["noul_pairs"]),
            "choice": choice_report(report["choice_rows"]),
            "by_phase": report["by_phase"],
        }
        del agent

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "scorecard.json"
    out_path.write_text(json.dumps(scorecard, indent=2, default=str) + "\n")
    print(json.dumps(scorecard["checkpoints"], indent=2, default=str))
    print(f"\nwrote {out_path}")
    print("NOTE: a fine-tune is a new confidence distribution -- refit temperatures and")
    print("thresholds against it (P4 pattern) before any gate trusts its numbers; never")
    print("carry the base's fitted thresholds over.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())