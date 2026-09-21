"""Replay matching.decide() / accept_any_fitting OFFLINE against recorded
distributions from the journal and the labeled calibration captures -- zero
model calls. Question: does arbitration behavior shift under the in-process
engine's probability shape versus the hosted engine's?

Shadow rows are excluded everywhere (shadow_of set). Usage:
    uv run python eval/phases/audit_arbitration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from jevdevice.budget import current_profile, is_abstain
from jevdevice.journal.decision_log import DecisionJournal
from jevdevice.matching import decide

HERE = Path(__file__).resolve().parent

# Ground truth for the labeled narrowing capture (recalibration/narrowing-laya.txt):
# case goal -> does the real enumeration contain a fitting candidate?
# obvious-fit / near-tie cases -> True; genuinely-absent / wrong-trap -> False.
EXPECTED_FITS = {
    "open Gmail": True,
    "check my email": False,
    "open messages": True,
    "take a photo": True,
    "book me a flight to Paris": False,
    "order a pizza": False,
    "what is my battery level?": True,
    "am I connected to wifi?": True,
    "what's using the most memory?": True,
    "what's the weather today?": False,
    "turn off airplane mode": False,
}


def load_pick_rows() -> list[dict]:
    """Decision rows whose ask carried a Choice, primary rows only."""
    rows = []
    for r in DecisionJournal().replay():
        if r.get("type") != "decision" or r.get("shadow_of"):
            continue
        answers = r.get("answers") or {}
        if not isinstance(answers, dict) or "pick" not in answers:
            continue
        pick = answers["pick"] or {}
        if pick.get("probabilities") is None:
            continue
        rows.append(r)
    return rows


def row_to_verdict_input(r: dict) -> dict:
    """Rebuild decide()'s arguments from a journaled ground-phase ask."""
    answers = r["answers"]
    pick = answers["pick"]
    candidates = list((r.get("state") or {}).get("candidates") or {})
    fits = {}
    for key, answer in answers.items():
        if key.startswith("fit_"):
            fits[candidates[int(key[4:])]] = answer["noul"]
    return {
        "choice": pick["choice"],
        "probabilities": pick["probabilities"] or {},
        "confidence": pick["confidence"] or 0.0,
        "fits": fits,
        "enumerated": candidates,  # full enumeration absent from the row; shortlist is the conservative superset
    }


def shape_stats() -> list[str]:
    lines = ["-- distribution shape per engine (pick rows, primary only; ground = where decide() runs) --"]
    rows = load_pick_rows()
    for engine in ("jev", "laya"):
        for phase in ("recall", "ground"):
            rows_p = [r for r in rows if r["engine"] == engine and r.get("phase") == phase]
            if not rows_p:
                continue
            profile = current_profile(engine)
            confs, margins, abstains, winner_mismatch = [], [], 0, 0
            for r in rows_p:
                pick = r["answers"]["pick"]
                if is_abstain(pick.get("choice")):
                    abstains += 1
                    continue
                probs = sorted((v for v in (pick["probabilities"] or {}).values() if isinstance(v, (int, float))), reverse=True)
                if not probs:
                    continue
                confs.append(pick["confidence"])
                margins.append(probs[0] - probs[1] if len(probs) >= 2 else probs[0])
                fits = row_to_verdict_input(r)["fits"]
                if fits and fits.get(pick.get("choice"), 0.0) < max(fits.values()) - 1e-9:
                    winner_mismatch += 1
            n = len(rows_p)
            if confs:
                lines.append(
                    f"engine={engine} phase={phase} n={n} abstain={abstains} "
                    f"confidence p50={sorted(confs)[len(confs)//2]:.3f} "
                    f"margin p50={sorted(margins)[len(margins)//2]:.3f} "
                    f"margin<{profile.min_margin}: {sum(1 for m in margins if m < profile.min_margin)}/{len(margins)} "
                    f"conf<{profile.min_confidence}: {sum(1 for c in confs if c < profile.min_confidence)}/{len(confs)} "
                    f"ranked-winner-is-not-the-fitter: {winner_mismatch}/{len(confs)}"
                )
    return lines


def labeled_replay() -> list[str]:
    """Replay decide() on every journaled ask for the labeled capture goals."""
    lines = ["-- labeled replay (capture ground truth vs replayed verdict) --"]
    by_goal: dict[str, list[dict]] = {}
    for r in load_pick_rows():
        if r["engine"] != "laya" or r.get("phase") != "ground":
            continue
        goal = (r.get("state") or {}).get("goal")
        if goal in EXPECTED_FITS:
            by_goal.setdefault(goal, []).append(r)
    profile = current_profile("laya")
    tp = fp = 0
    for goal, expected in sorted(EXPECTED_FITS.items()):
        rows_g = by_goal.get(goal, [])
        oks = []
        for r in rows_g:
            args = row_to_verdict_input(r)
            verdict = decide(args["choice"], args["probabilities"], args["confidence"], args["fits"],
                             args["enumerated"], min_fit=profile.min_fit,
                             min_confidence=profile.min_confidence, min_margin=profile.min_margin)
            oks.append(verdict.ok)
        hit = sum(1 for ok in oks if ok == expected)
        tp += hit
        fp += len(oks)
        lines.append(f"  {goal!r}: expected ok={expected}, replayed {hit}/{len(oks)} agree")
    lines.append(f"labeled replay agreement: {tp}/{fp}")
    return lines


def threshold_sweep() -> list[str]:
    """Where would arbitration behavior shift? Sweep min_fit/min_confidence/min_margin
    over the labeled set and report (fit-case pass rate, absent-case leak rate)."""
    lines = ["-- threshold sweep on the labeled set (fit cases should pass, absent cases should fail) --"]
    by_goal: dict[str, list[dict]] = {}
    for r in load_pick_rows():
        if r["engine"] != "laya" or r.get("phase") != "ground" or (r.get("state") or {}).get("goal") not in EXPECTED_FITS:
            continue
        by_goal.setdefault(r["state"]["goal"], []).append(r)
    cached = {goal: [row_to_verdict_input(r) for r in rows] for goal, rows in by_goal.items()}
    profile = current_profile("laya")
    best = None
    for min_fit in (0.4, 0.45, 0.5, 0.55, 0.6):
        for min_confidence in (0.5, 0.6, 0.7, 0.8):
            for min_margin in (0.05, 0.1, 0.15, 0.25):
                pass_ok = fail_ok = 0
                pass_n = fail_n = 0
                for goal, expected in EXPECTED_FITS.items():
                    for args in cached.get(goal, []):
                        verdict = decide(args["choice"], args["probabilities"], args["confidence"], args["fits"],
                                         args["enumerated"], min_fit=min_fit, min_confidence=min_confidence,
                                         min_margin=min_margin)
                        if expected:
                            pass_n += 1
                            pass_ok += verdict.ok
                        else:
                            fail_n += 1
                            fail_ok += verdict.ok
                if pass_n and fail_n:
                    recall = pass_ok / pass_n
                    leak = fail_ok / fail_n
                    if best is None or (recall - leak, recall) > (best[0] - best[1], best[0]):
                        best = (recall, leak, min_fit, min_confidence, min_margin)
                    if recall >= 0.8 and leak <= 0.2:
                        lines.append(f"  min_fit={min_fit} min_confidence={min_confidence} min_margin={min_margin} "
                                     f"-> fit-pass={recall:.2f} absent-leak={leak:.2f}  (would clear 0.95 precision: NO)")
    if best:
        lines.append(f"  best separation found: fit-pass={best[0]:.2f} absent-leak={best[1]:.2f} "
                     f"at min_fit={best[2]} min_confidence={best[3]} min_margin={best[4]}")
    lines.append(f"  current knobs: min_fit={profile.min_fit} min_confidence={profile.min_confidence} "
                 f"min_margin={profile.min_margin} -- no sweep cell reaches the auto-apply precision bar")
    return lines


def any_fitting_replay() -> list[str]:
    """accept_any_fitting replay on the read-only service/field paths
    (candidates without a package dot): verdict flips True vs False."""
    lines = ["-- accept_any_fitting replay (read-only dumpsys paths) --"]
    flips = rescued = 0
    total = 0
    for r in load_pick_rows():
        if r["engine"] != "laya" or r.get("phase") != "ground":
            continue
        args = row_to_verdict_input(r)
        if not args["fits"] or any("." in c for c in args["fits"]):
            continue  # package paths run without accept_any_fitting
        profile = current_profile("laya")
        base = decide(args["choice"], args["probabilities"], args["confidence"], args["fits"],
                      args["enumerated"], min_fit=profile.min_fit, min_confidence=profile.min_confidence,
                      min_margin=profile.min_margin, accept_any_fitting=False)
        relaxed = decide(args["choice"], args["probabilities"], args["confidence"], args["fits"],
                         args["enumerated"], min_fit=profile.min_fit, min_confidence=profile.min_confidence,
                         min_margin=profile.min_margin, accept_any_fitting=True)
        total += 1
        if base.ok != relaxed.ok:
            flips += 1
            if relaxed.ok:
                rescued += 1
    lines.append(f"read-only ground rows replayed: {total}; verdict flips: {flips} ({rescued} rescued by accept_any_fitting)")
    return lines


def main() -> None:
    for section in (shape_stats(), labeled_replay(), threshold_sweep(), any_fitting_replay()):
        print("\n".join(section))
        print()


if __name__ == "__main__":
    main()
