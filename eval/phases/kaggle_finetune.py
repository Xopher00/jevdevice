"""Supervised fine-tune of laya-typed-decisions on the journal export.

Runs on Kaggle (2xT4, ~4-5 h reference) -- NOTHING in this script touches the
local device path; it consumes eval/phases/finetune/{train,val}.jsonl from the
dev-half journal export only. The sacred goals.yaml heldout half is never read.

Input rows: {"state", "questions" (wire dicts), "supervision", ...}. Each row
is exploded into ONE training sequence per supervised question -- laya's
build_sequence encodes a single (state, question) pair per forward pass, which
is exactly what Agent.predict does at inference, so train and serve encodings
match byte-for-byte (build_sequence comes from the laya package itself).

Supervision -> target marker index:
  noul   -> options are always [false, true]; target = 1 if label else 0
  choice -> target = index of correct_option among the wire criteria keys
  score  -> target = the verified level index (no examples exist yet; wired
            for when they do)

Class imbalance is real (fit_noul positives ~6%): POS_WEIGHT_NOUL upweights
the positive class in the noul CE. Val loss (not train accuracy) drives early
stop; per-type accuracy/Brier are reported each pass.

Output: a full checkpoint directory (base layout preserved: rl_agent_config.json,
tokenizer/, model.safetensors replaced) so laya's Agent loads it directly.
Publish: upload the directory to its own HF repo, pin the revision, and point
the runtime at it with JEV_LAYA_REVISION=<revision>. Recalibration must rerun
against the new artifact before any threshold is trusted.

Kaggle setup:
  !pip install -q laya
  !python /kaggle/input/<dataset>/kaggle_finetune.py --train ... --val ...

Run locally (CUDA): uv run python eval/phases/kaggle_finetune.py \
    --train eval/phases/finetune/train.jsonl --val eval/phases/finetune/val.jsonl \
    --base laya-typed-decisions --out eval/phases/finetune/checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from laya.common import (  # type: ignore[import-untyped]
    QTYPES,
    build_model,
    build_sequence,
)
from torch.nn import functional as F

# named knobs (all overridable by CLI; defaults recorded in the scorecard)
LR = 2e-5
EPOCHS = 3
BATCH_SIZE = 16
GRAD_ACCUM = 4
POS_WEIGHT_NOUL = 4.0     # fit_noul positives are rare; >1 upweights them
LABEL_SMOOTHING = 0.0
VAL_EVERY = 0.25          # fraction of an epoch between validations
PATIENCE = 4              # validations without val-loss improvement -> stop
MAX_GRAD_NORM = 1.0
SEED = 20260921
WEIGHT_DECAY = 0.01


def load_checkpoint_dir(spec: str, token: str | None) -> Path:
    """Base checkpoint -> local directory (HF hub or already-local path)."""
    if Path(spec).exists():
        return Path(spec)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download("convaiinnovations/laya", revision=spec, allow_patterns=["typed-decisions/*"], token=token))


def wire_to_internal(question: dict) -> dict:
    """The frozen wire dict -> laya's internal {t, ins, crit} form (the same
    conversion Agent.predict performs; criteria ride through unchanged --
    render_criterion handles structured values)."""
    return {"t": QTYPES[question["type"]], "ins": question["instructions"], "crit": question.get("criteria")}


def target_index(question: dict, supervision: dict) -> int | None:
    """Marker index of the ground-truth option for one supervised question."""
    if supervision["type"] == "noul":
        return 1 if supervision["label"] else 0  # noul options are always [false, true]
    if supervision["type"] == "choice":
        correct = supervision["correct_option"]
        options = list(question["criteria"])
        try:
            return options.index(correct)
        except ValueError:
            return None  # the option set cannot express the answer: skip
    if supervision["type"] == "score":
        level = supervision.get("level")
        return None if level is None else int(level)
    return None


def build_items(path: Path, tok, cfg: dict) -> list[dict]:
    """JSONL export -> per-question training items (encoded once, reused)."""
    max_len = int(cfg.get("max_len", 1024))
    head_max_len = int(cfg.get("head_max_len", 192))
    items: list[dict] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for qname, supervision in row["supervision"].items():
            question = row["questions"][qname]
            index = target_index(question, supervision)
            if index is None:
                skipped += 1
                continue
            ids, markers = build_sequence(tok, row["state"], wire_to_internal(question), max_len=max_len, head_max_len=head_max_len)
            items.append({"ids": ids, "markers": markers, "target": index, "type": supervision["type"],
                          "goal": row.get("goal"), "phase": row.get("phase")})
    print(f"{path.name}: {len(items)} items ({skipped} unscoreable supervision entries skipped)")
    return items


def collate(batch: list[dict], pad_id: int):
    """Pad to the longest sequence in the batch; keep marker positions."""
    width = max(len(item["ids"]) for item in batch)
    input_ids, mask, marker_pos, marker_mask, targets, qtype = [], [], [], [], [], []
    for item in batch:
        n = len(item["ids"])
        input_ids.append(item["ids"] + [pad_id] * (width - n))
        mask.append([1] * n + [0] * (width - n))
        pos = item["markers"] + [0] * (8 - len(item["markers"]))
        present = [1] * len(item["markers"]) + [0] * (8 - len(item["markers"]))
        marker_pos.append(pos[:8])
        marker_mask.append(present[:8])
        targets.append(item["target"])
        qtype.append(item["internal_t"])
    return (torch.tensor(input_ids), torch.tensor(mask), torch.tensor(marker_pos),
            torch.tensor(marker_mask), torch.tensor(targets), torch.tensor(qtype))


def loss_for(logits: torch.Tensor, targets: torch.Tensor, qtype_onehot_noul: torch.Tensor) -> torch.Tensor:
    """CE over marker logits; noul positives upweighted by POS_WEIGHT_NOUL."""
    per_example = F.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING, reduction="none")
    noul = qtype_onehot_noul.bool()
    positive = (targets == 1) & noul
    weights = torch.ones_like(per_example)
    weights[positive] = POS_WEIGHT_NOUL
    return (per_example * weights).sum() / weights.sum()


@torch.no_grad()
def evaluate(model, batches, device) -> dict:
    model.eval()
    per_type: dict[str, list[tuple[float, int]]] = defaultdict(list)
    total_loss, n_batches = 0.0, 0
    for batch in batches:
        input_ids, mask, marker_pos, marker_mask, targets, qtype = [t.to(device) for t in batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids, mask, marker_pos, marker_mask, qtype)
        total_loss += F.cross_entropy(logits.float(), targets, reduction="mean").item()
        n_batches += 1
        predictions = logits.argmax(-1).tolist()
        for prediction, target, qt in zip(predictions, targets.tolist(), qtype.tolist()):
            name = {v: k for k, v in QTYPES.items()}.get(qt, "choice")
            per_type[name].append((1.0 if prediction == target else 0.0, target))
    accuracy = {name: sum(s for s, _ in rows) / len(rows) for name, rows in per_type.items()}
    return {"loss": total_loss / max(1, n_batches), "accuracy": accuracy,
            "n": {name: len(rows) for name, rows in per_type.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--base", default="typed-decisions",
                        help="HF revision/subfolder of the base checkpoint (e.g. typed-decisions)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--pos-weight-noul", type=float, default=POS_WEIGHT_NOUL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    base_dir = load_checkpoint_dir(args.base, args.hf_token)
    cfg = json.loads((base_dir / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(base_dir / "tokenizer")

    model = build_model(cfg, encoder_dir=str(base_dir / "encoder"))
    weights = load_file(base_dir / "model.safetensors")
    model.load_state_dict(weights)
    model.to(args.device)

    train_items = build_items(args.train, tok, cfg)
    val_items = build_items(args.val, tok, cfg)
    # the qtype the model consumes (QTYPES): 0 choice, 1 score, 2 noul -- the
    # supervision kind IS the wire question's type by construction
    for item in train_items + val_items:
        item["internal_t"] = QTYPES[item["type"]]

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.sep_token_id

    def batches(items: list[dict], shuffle: bool):
        order = list(range(len(items)))
        if shuffle:
            random.shuffle(order)
        for start in range(0, len(order), args.batch):
            yield collate([items[i] for i in order[start:start + args.batch]], pad_id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_items) / args.batch / args.accum)
    validate_every = max(1, int(steps_per_epoch * VAL_EVERY))
    best_val, stale, step, epoch = math.inf, 0, 0, 0
    started = time.time()
    scorecard = {"knobs": vars(args) | {"VAL_EVERY": VAL_EVERY, "PATIENCE": PATIENCE,
                                        "MAX_GRAD_NORM": MAX_GRAD_NORM, "WEIGHT_DECAY": WEIGHT_DECAY},
                 "n_train_items": len(train_items), "n_val_items": len(val_items), "history": []}

    while epoch < args.epochs and stale < PATIENCE:
        model.train()
        optimizer.zero_grad()
        for index, batch in enumerate(batches(train_items, shuffle=True)):
            input_ids, mask, marker_pos, marker_mask, targets, qtype = [t.to(args.device) for t in batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(input_ids, mask, marker_pos, marker_mask, qtype)
            loss = loss_for(logits.float(), targets, qtype == QTYPES["noul"]) / args.accum
            loss.backward()
            if (index + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % validate_every == 0 or index + 1 >= steps_per_epoch:
                    metrics = evaluate(model, list(batches(val_items, shuffle=False)), args.device)
                    scorecard["history"].append({"epoch": epoch, "step": step, "val": metrics,
                                                 "minutes": round((time.time() - started) / 60, 1)})
                    print(f"epoch {epoch} step {step}: val_loss={metrics['loss']:.4f} "
                          f"acc={json.dumps(metrics['accuracy'])} ({scorecard['history'][-1]['minutes']} min)")
                    if metrics["loss"] < best_val:
                        best_val, stale = metrics["loss"], 0
                        out_dir = args.out
                        if out_dir.exists():
                            shutil.rmtree(out_dir)
                        shutil.copytree(base_dir, out_dir)
                        save_file({k: v for k, v in model.state_dict().items()}, str(out_dir / "model.safetensors"))
                    else:
                        stale += 1
        epoch += 1

    (args.out / "finetune_scorecard.json").write_text(json.dumps(scorecard, indent=2, default=str) + "\n")
    print(f"done: best val_loss={best_val:.4f}; checkpoint written to {args.out}")
    print("publish: upload the directory to its own HF repo, pin the commit, "
          "then set JEV_LAYA_REVISION=<revision> and rerun recalibration against it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())