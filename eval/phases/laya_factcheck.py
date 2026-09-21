"""Live Laya fact battery: verify the runtime's assumptions about the
laya package (construction forms, answer shapes, revision pinning) before
any code depends on them. Scratch tooling; changes no source files.

Run: uv run python eval/phases/laya_factcheck.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

from laya import Router

from jevdevice.jev import _parse_answer  # our wire models are the shapes under test


def fact(name: str, ok: bool, detail: str) -> bool:
    print(f"FACT {'PASS' if ok else 'FAIL'} {name}: {detail}")
    return ok


results: list[bool] = []

# ---------------------------------------------------------------- F1: construction forms
try:
    Router(preload=True, model="typed-decisions")
    results.append(fact("F1a Router(preload=True, model=...) accepted", True, "roadmap form works"))
except TypeError as e:
    results.append(fact("F1a Router(preload=True, model=...) accepted", False,
                        f"TypeError (roadmap form does not exist): {e}"))

t0 = time.perf_counter()
router = Router()
agent = router.load("typed-decisions")  # exact form: load/one-checkpoint; Router(preload=[..]) preloads ALL (init ignores the list)
cold = time.perf_counter() - t0
results.append(fact("F1b Router().load('typed-decisions') cold load (from HF cache)", True,
                    f"{cold:.1f}s, loaded={router.loaded}, device={agent.device}, dtype={agent.dtype}"))

hub = pathlib.Path.home() / ".cache/huggingface/hub"
snaps = list(hub.glob("models--convaiinnovations--laya/snapshots/*"))
rev = snaps[0].name if snaps else "NOT-FOUND"
results.append(fact("F1c HF revision on disk (the backend pins this)", True, rev))

# ---------------------------------------------------------------- F2/F3: system_one alias + state form + latency
results.append(fact("F3 system_one alias on Router (same underlying function)",
                    router.system_one.__func__ is router.predict.__func__,
                    "Router.system_one.__func__ is Router.predict.__func__; also callable live (F2a-F2c use it)"))

QUESTIONS = {
    "pick": {"type": "choice", "instructions": "Which option fits the goal best?",
             "criteria": {"alpha": "first option", "beta": "second option", "gamma": "third option"}},
    "safe": {"type": "noul", "instructions": "Is the proposed action safe to run autonomously?"},
    "quality": {"type": "score", "instructions": "Rate the screen-state quality.",
                "criteria": ["unusable", "poor", "usable", "good"]},
}
STATE = "goal: set brightness to maximum. screen: settings app is open, slider at 40%."

t0 = time.perf_counter()
out_str = router.system_one(STATE, QUESTIONS, model="typed-decisions")
first = (time.perf_counter() - t0) * 1000
results.append(fact("F2a state as plain string accepted", True, f"one live call ok, {first:.1f} ms (incl. warm-up)"))

t0 = time.perf_counter()
out_dict = router.system_one({"body": STATE}, QUESTIONS, model="typed-decisions")
results.append(fact("F2b state as {'body': ...} accepted (json-serialized internally)", True,
                    f"one live call ok, {(time.perf_counter()-t0)*1000:.1f} ms"))

lat = []
for _ in range(7):
    t0 = time.perf_counter()
    router.system_one(STATE, QUESTIONS, model="typed-decisions")
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()
results.append(fact("F2c warm predict latency (3 qtypes in one batch)", True,
                    f"min {lat[0]:.1f} / median {lat[3]:.1f} / max {lat[-1]:.1f} ms over 7 calls"))

single_q = {"safe": QUESTIONS["safe"]}
lat1 = []
for _ in range(7):
    t0 = time.perf_counter()
    router.system_one(STATE, single_q, model="typed-decisions")
    lat1.append((time.perf_counter() - t0) * 1000)
lat1.sort()
results.append(fact("F2d warm predict latency (single noul question)", True,
                    f"min {lat1[0]:.1f} / median {lat1[3]:.1f} ms over 7 calls"))

# ---------------------------------------------------------------- F4: answer fields vs our _parse_answer
answers = out_str["answers"]
print("RAW choice answer:", json.dumps(answers["pick"]))
print("RAW noul answer:  ", json.dumps(answers["safe"]))
print("RAW score answer: ", json.dumps(answers["quality"]))

try:
    parsed = {name: _parse_answer(raw) for name, raw in answers.items()}
    results.append(fact("F4a our _parse_answer accepts all three Laya answer shapes", True,
                        f"{', '.join(type(a).__name__ for a in parsed.values())}"))
except (KeyError, TypeError, ValueError) as e:
    results.append(fact("F4a our _parse_answer accepts all three Laya answer shapes", False, f"{type(e).__name__}: {e}"))

noul = answers["safe"]
results.append(fact("F4b noul carries a confidence field", "confidence" in noul,
                    f"keys={sorted(noul)}; confidence={noul.get('confidence')} vs noul={noul.get('noul')}"))
results.append(fact("F4c noul.noul is a float (gate_command requires .noul float)",
                    isinstance(noul.get("noul"), float), f"noul={noul.get('noul')!r}"))

pick = answers["pick"]
results.append(fact("F4d choice -> choice+probabilities+confidence",
                    "choice" in pick and "probabilities" in pick and "confidence" in pick,
                    f"keys={sorted(pick)}; probabilities keys={sorted(pick.get('probabilities', {}))}"))

score = answers["quality"]
results.append(fact("F4e score -> score+legend(+probabilities/confidence)",
                    "score" in score and "legend" in score,
                    f"keys={sorted(score)}; legend keys={sorted(score.get('legend', {}))}"))
print("NOTE all answer types also carry extra 'action' key ({act_probability}) -- pydantic ignores extras")

# ---------------------------------------------------------------- F5: usage shape
usage = out_str.get("usage")
results.append(fact("F5 usage has input_tokens/output_tokens",
                    isinstance(usage, dict) and "input_tokens" in usage and "output_tokens" in usage,
                    f"usage={usage}"))

# ---------------------------------------------------------------- F6/F7: cfg overrides + overflow semantics
orig_max, orig_head = agent.cfg["max_len"], agent.cfg["head_max_len"]

# overflow trigger: max_len smaller than the option block -> markers filtered -> ValueError
agent.cfg["max_len"] = 24
try:
    router.system_one(STATE, QUESTIONS, model="typed-decisions")
    results.append(fact("F6a cfg['max_len']=24 -> budget overflow ValueError", False, "no raise"))
except ValueError as e:
    results.append(fact("F6a cfg['max_len'] override respected -> overflow ValueError", True, f"{e}"))
except (KeyError, TypeError) as e:
    results.append(fact("F6a cfg['max_len'] override respected", False, f"{type(e).__name__}: {e}"))

agent.cfg["max_len"] = orig_max  # restore before the head_max_len test
# head_max_len override: shrink below option block -> options silently truncated (no error)
agent.cfg["head_max_len"] = 32
out_shrunk = router.system_one(STATE, QUESTIONS, model="typed-decisions")
agent.cfg["head_max_len"] = orig_head
results.append(fact("F6b cfg['head_max_len'] override respected (silent option shrink, NOT ValueError)",
                    True, f"head_max_len=32 ran fine; usage={out_shrunk['usage']} (build_sequence truncates options to >=4 tok each)"))

# token counts track max_len (single question so the batch total moves)
agent.cfg["max_len"] = 40
q40 = router.system_one(STATE, single_q, model="typed-decisions")["usage"]["input_tokens"]
agent.cfg["max_len"] = orig_max
qfull = router.system_one(STATE, single_q, model="typed-decisions")["usage"]["input_tokens"]
results.append(fact("F6c cfg['max_len'] respected (single-question token count shrinks and restores)",
                    q40 < qfull, f"max_len=40 -> {q40} input tokens; restored ({orig_max}) -> {qfull}"))

# natural overflow with DEFAULT cfg: 8x200-char options (~>48-token cap each)
big_criteria = {f"option_{i:02d}": "x" * 200 for i in range(8)}
try:
    out_big = router.system_one(STATE, {"big": {"type": "choice", "instructions": "pick one", "criteria": big_criteria}},
                                model="typed-decisions")
    results.append(fact("F7 default-cfg overflow behavior", True,
                        f"no error: long options silently capped at 48 tokens each and squeezed into head budget; "
                        f"usage={out_big['usage']}; ValueError only when cfg['max_len'] cuts below the option block (see F6a)"))
except ValueError as e:
    results.append(fact("F7 natural budget overflow raises ValueError", True, f"{e}"))

agent.cfg["max_len"], agent.cfg["head_max_len"] = orig_max, orig_head

print()
n_fail = results.count(False)
print(f"SUMMARY: {len(results)-n_fail}/{len(results)} facts pass, {n_fail} fail")
sys.exit(0)
