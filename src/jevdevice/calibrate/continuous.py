"""Reporting + tighten-only refit over the core journal's calibration units:
zero model calls, offline. `run()` prints per-unit probability quality
(accuracy/Brier/log-loss/ECE) and a conformal/ACI spike, then refits every
unit through core's `recalibrate()` (tighten-only unless `human_signoff`).

Run: uv run python -m jevdevice.calibrate.continuous [journal_dir] [--apply]
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from typesymbolic.calibrate import recalibrate
from typesymbolic.journal import DECISION, VERDICT, Journal
from typesymbolic.vocab import unit_name

from jevdevice.budget import ENGINE_ENV, JEV_ENGINE_NAME, profile_for
from jevdevice.calibrate.units import CALIBRATION_UNITS, store
from jevdevice.journal import decision_log

ENV_PROMOTION_SIGNOFF = "JEV_PROMOTION_SIGNOFF"
CONFORMAL_ALPHA = 0.05
ACI_GAMMA = 0.01
ECE_BINS = 10
MIN_LABELS = 20


def signoff_from_env() -> bool:
    """The human-set promotion switch core's `recalibrate` needs explicitly."""
    return os.environ.get(ENV_PROMOTION_SIGNOFF, "").strip().lower() in {"1", "yes", "true", "signoff"}


def resolve_engine(engine: str | None) -> str:
    """--engine flag > JEV_ENGINE env > jev -- same precedence as bootstrap()."""
    if engine:
        return engine
    return (os.environ.get(ENGINE_ENV) or JEV_ENGINE_NAME).strip().lower()


@dataclass(frozen=True)
class KnobResult:
    """One knob's refit outcome, shaped for a caller (the demo) to render."""

    knob: str
    group: str
    scale: str
    n_labels: int
    before: float
    after: float
    applied: bool
    note: str


def ece(values: list[float], labels: list[bool], bins: int = ECE_BINS) -> float:
    total, error = len(labels), 0.0
    if not total:
        return float("nan")
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(values) if (lo <= p < hi) or (b == bins - 1 and p == hi)]
        if idx:
            mean_p = sum(values[i] for i in idx) / len(idx)
            mean_y = sum(1 for i in idx if labels[i]) / len(idx)
            error += len(idx) / total * abs(mean_p - mean_y)
    return error


def brier(values: list[float], labels: list[bool]) -> float:
    return sum((p - y) ** 2 for p, y in zip(values, labels)) / len(labels)


def logloss(values: list[float], labels: list[bool], eps: float = 1e-6) -> float:
    clipped = [min(max(p, eps), 1 - eps) for p in values]
    return -sum(math.log(p if y else 1 - p) for p, y in zip(clipped, labels)) / len(labels)


def fit_temperature(vectors: list[tuple[dict[str, float], str]]) -> float | None:
    """Best T (0.10..3.00) for the p^(1/T) NLL over choice vectors;
    reporting diagnostic only, never applied to a gate or the runtime."""
    def nll(t: float) -> float:
        scaled = ({k: p ** (1 / t) for k, p in probs.items()} for probs, _ in vectors)
        return sum(-math.log(s[c] / sum(s.values())) if c in s and sum(s.values()) > 0 else math.inf
                   for s, (_, c) in zip(scaled, vectors)) / len(vectors)
    return min((round(0.05 * i, 2) for i in range(2, 61)), key=nll) if vectors else None


def temperature_vectors(journal: Journal, *, engine: str) -> list[tuple[dict[str, float], str]]:
    """(probabilities, correct_option) for verified choice answers: decision
    rows joined to verdict rows on call_id, scale='confidence' only."""
    rows = list(journal.replay())
    decisions = {row["call_id"]: row for row in rows
                 if row.get("type") == DECISION and row.get("engine") == engine}
    vectors: list[tuple[dict[str, float], str]] = []
    for row in rows:
        if row.get("type") != VERDICT or row.get("status") != "verified" or row.get("calibrate") is False:
            continue
        decision = decisions.get(row.get("call_id")) or {}
        for key in row.get("tests") or []:
            ref = (decision.get("questions") or {}).get(key) or {}
            answer = (decision.get("answers") or {}).get(key) or {}
            if ref.get("scale") == "confidence" and answer.get("probabilities") and answer.get("choice"):
                vectors.append((answer["probabilities"], answer["choice"]))
    return vectors


def conformal_threshold(values: list[float], labels: list[bool], *, alpha: float = CONFORMAL_ALPHA) -> float | None:
    """Loosest threshold whose empirical bad-outcome rate stays <= alpha."""
    for t in [round(0.05 * i, 2) for i in range(1, 20)]:
        approved = [y for p, y in zip(values, labels) if p >= t]
        if approved and sum(1 for y in approved if not y) / len(approved) <= alpha:
            return t
    return None


def aci_update(values: list[float], labels: list[bool], *, alpha: float = CONFORMAL_ALPHA, gamma: float = ACI_GAMMA) -> dict:
    """Online alpha_t simulation over the labels in order: approve when the
    value clears the (1-alpha_t) empirical quantile seen so far."""
    alpha_t, approved, bad = alpha, 0, 0
    seen: list[float] = []
    for value, label in zip(values, labels):
        seen.append(value)
        ranked = sorted(seen)
        index = min(len(ranked) - 1, max(0, math.ceil((1 - alpha_t) * len(ranked)) - 1))
        if value >= ranked[index]:
            approved += 1
            err = 0 if label else 1
            bad += err
            alpha_t = min(1.0, max(0.0, alpha_t + gamma * (alpha - err)))
    return {"n": len(values), "approved": approved, "bad": bad,
            "realized_bad_rate": bad / approved if approved else None, "alpha_final": round(alpha_t, 4)}


def conformal_spike(values: list[float], labels: list[bool], *, alpha: float = CONFORMAL_ALPHA) -> dict:
    """Would a conformal threshold + ACI beat the fitted point on this
    sample? Adopt only if the realized bad rate clears the bar with enough
    approval mass; otherwise 'keep', with the reason recorded."""
    n = len(labels)
    threshold = conformal_threshold(values, labels, alpha=alpha)
    approved = [y for p, y in zip(values, labels) if threshold is not None and p >= threshold]
    realized = (sum(1 for y in approved if not y) / len(approved)) if approved else None
    reasons = []
    if threshold is None:
        reasons.append(f"no threshold reaches bad-rate <= {alpha}")
    elif realized is not None and realized > alpha:
        reasons.append(f"realized bad rate {realized:.3f} > {alpha}")
    elif len(approved) < MIN_LABELS:
        reasons.append(f"approval mass {len(approved)} < {MIN_LABELS} -- noise-thin")
    elif n < MIN_LABELS:
        reasons.append(f"n={n} < {MIN_LABELS} -- keep collecting")
    return {"n": n, "conformal_threshold": threshold, "approved_at_threshold": len(approved),
            "realized_bad_rate": realized, "aci": aci_update(values, labels, alpha=alpha),
            "verdict": "keep" if reasons else "adopt", "reasons": reasons}


def report_unit(name: str, pairs: list[tuple[float, bool]], *, quiet: bool = False) -> dict:
    """Probability-quality report for one calibration unit's labeled pairs."""
    say = (lambda *_a: None) if quiet else print
    if not pairs:
        say(f"\n== {name}: no labeled rows ==")
        return {"n": 0}
    values, labels = [p for p, _ in pairs], [y for _, y in pairs]
    n = len(labels)
    metrics = {
        "n": n, "accuracy@0.5": sum((p >= 0.5) == y for p, y in zip(values, labels)) / n,
        "brier": brier(values, labels), "logloss": logloss(values, labels), "ece": ece(values, labels),
    }
    say(f"\n== {name} == n={n}")
    say("  metrics: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
    if n < MIN_LABELS:
        say(f"  WARN: n={n} < {MIN_LABELS} -- thin sample, keep collecting")
    spike = conformal_spike(values, labels)
    say(f"  conformal/ACI: threshold={spike['conformal_threshold']} approved={spike['approved_at_threshold']} "
        f"bad_rate={spike['realized_bad_rate']} -> {spike['verdict']} {spike['reasons']}")
    metrics["conformal_spike"] = spike
    return metrics


def run(
    journal_dir: Path | None = None, *, engine: str | None = None, human_signoff: bool | None = None,
    calibration_store=None, quiet: bool = False,
) -> dict[str, KnobResult]:
    """One refit pass: report each calibration unit, then recalibrate it
    through core (tighten-only unless `human_signoff`). Zero model calls.
    `engine=None` resolves --engine-flag precedence (see `resolve_engine`)."""
    say = (lambda *_a: None) if quiet else print
    engine = resolve_engine(engine)
    directory = journal_dir or decision_log.journal_dir()
    journal = Journal(root=directory, rotation="daily", background_writes=False)
    profile = profile_for(engine)
    signoff = signoff_from_env() if human_signoff is None else human_signoff
    calibration_store = calibration_store or store()
    say(f"== continuous calibration: engine={engine} journal={directory}")

    results: dict[str, KnobResult] = {}
    for knob, (group, scale) in CALIBRATION_UNITS.items():
        default = getattr(profile, knob)
        pairs = journal.labeled_pairs(group, scale, engine=engine, any_revision=True)
        report_unit(f"{group}|{scale}", pairs, quiet=quiet)
        # before: store's current value, read ahead of recalibrate's own write.
        before = calibration_store.get(unit_name(group, scale), engine=engine, model_revision=None, default=default)
        result = recalibrate(
            journal=journal, store=calibration_store, group=group, scale=scale, engine=engine,
            default_threshold=default, human_signoff=signoff, pool_revisions=True,
        )
        results[knob] = KnobResult(
            knob=knob, group=group, scale=scale, n_labels=len(pairs),
            before=before, after=result.threshold, applied=result.applied, note=result.note,
        )
        direction = "unchanged" if result.threshold == default and not result.applied else (
            "applied" if result.applied else "refused/kept"
        )
        say(f"[{knob}] incumbent={default} proposed={result.proposal.proposed} "
            f"-> {direction} threshold={result.threshold} ({result.note})")
    fitted_t = fit_temperature(temperature_vectors(journal, engine=engine))
    say(f"[diagnostic, not applied] fitted temperature: {fitted_t if fitted_t is not None else 'n/a'}")
    return results


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    apply = "--apply" in argv
    argv = [a for a in argv if a != "--apply"]
    engine = None
    if "--engine" in argv:
        i = argv.index("--engine")
        engine = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    directory = Path(argv[0]) if argv else None
    results = run(directory, engine=engine, human_signoff=signoff_from_env() if apply else False)
    applied = {k: r for k, r in results.items() if r.applied}
    if apply and applied:
        print("PROMOTED: " + ", ".join(f"{k}={r.after}" for k, r in applied.items()))
    elif not apply:
        # recalibrate() always writes tightenings to the store; only a
        # loosening proposal is refused here (human_signoff stays False).
        print("shadow-run mode: --apply not passed -- tightenings still promote; only loosening is refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
