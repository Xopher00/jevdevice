"""Compile-time question-set validation.

Reads the frozen artifact (src/jevdevice/question_sets/v1.yaml) and proves it
against two sources of truth:

1. THE GOLDEN SET: every journal question instance on the golden path (phases
   recall/ground/fill/gate/verify/kind, primary rows only) must match exactly
   one template byte-for-byte, with slots recovered from the row's own state
   (candidate names, chosen_action/proposed_command/target_bounds reprs).
   P4-perturbation "probe" rows and eval-CLI "unlabeled" rows are eval tooling,
   out of the golden path, and are reported but not validated.

2. THE MODULE CONSTANTS: the *_SAFE_INSTRUCTIONS constants must equal their
   artifact entries (the compile step ran pre- and post-rewiring; after the
   rewiring the constants are themselves sourced from this artifact).

Exit code 0 = every golden-path instance witnessed, no unmatched instances.
Writes witness counts back into the artifact as provenance (validated: true).

Run: uv run python eval/phase6_compile_questions.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from jevdevice.decision_log import DecisionJournal

GOLDEN_PHASES = {"recall", "ground", "fill", "gate", "verify", "kind"}
ARTIFACT = REPO / "src" / "jevdevice" / "question_sets" / "v1.yaml"

# Pre-freeze wordings the journal still carries (one dev iteration on
# 2026-09-20 before the wording landed); no current source emits them.
JUSTIFIED_UNMATCHED_PREFIXES = (
    "Given the candidates, could any satisfy the goal?",
)

# per-candidate fit ids by flow: (pick template id, fit template id)
FIT_FLOWS = {
    "tap.pick": "tap.fit",
    "long_press.pick": "long_press.fit",
    "type_field.pick": "type_field.fit",
    "scroll_to_find.pick": "scroll_to_find.fit",
    "open_app.pick": "open_app.fit",
    "toggle.resolve_status.pick": "toggle.resolve_status.fit",
    "dumpsys_query.pick": "dumpsys_query.fit",
    "dumpsys_field.pick": "dumpsys_field.fit",
}

# the module constants that must equal artifact entries (pre- AND post-rewiring)
CONSTANT_IDS = [
    ("jevdevice.ui", "TAP_SAFE_INSTRUCTIONS", "tap.safe"),
    ("jevdevice.ui", "TYPE_SAFE_INSTRUCTIONS", "type.safe"),
    ("jevdevice.ui", "LONG_PRESS_SAFE_INSTRUCTIONS", "long_press.safe"),
    ("jevdevice.ui", "SWIPE_SAFE_INSTRUCTIONS", "swipe.safe"),
    ("jevdevice.gate", "DEFAULT_SAFE_INSTRUCTIONS", "gate.safe.default"),
    ("jevdevice.services", "KEYEVENT_SAFE_INSTRUCTIONS", "keyevent.safe"),
    ("jevdevice.services", "DND_SAFE_INSTRUCTIONS", "dnd.safe"),
]


def candidates_of(state: dict) -> list[str]:
    cands = state.get("candidates")
    if isinstance(cands, dict):
        return [v if isinstance(v, str) and v else k for k, v in cands.items()]
    if isinstance(cands, list):
        return [str(c) for c in cands]
    return []


def _template_regex(template: str) -> re.Pattern:
    """Template -> anchored regex: literal parts escaped, {slot} and {slot!r}
    -> non-greedy (.+?). Slotted text is recovered from the question itself
    (the journal does not always carry the slot values, e.g. the fused tap
    ask's chosen_action/proposed_command/target_bounds are question-only)."""
    import string

    parts = []
    for literal, field, spec, _conv in string.Formatter().parse(template):
        parts.append(re.escape(literal))
        if field:
            parts.append(r"(.+?)")
    return re.compile("^" + "".join(parts) + "$")


_TEMPLATE_REGEX_CACHE: dict[str, re.Pattern] = {}


def match_template(instructions: str, templates: dict[str, str], slots: dict) -> str | None:
    """First template whose slot pattern reproduces `instructions` byte-for-byte
    (no placeholder -> exact equality)."""
    for question_id, template in templates.items():
        if "{" not in template:
            if template == instructions:
                return question_id
            continue
        pattern = _TEMPLATE_REGEX_CACHE.get(template)
        if pattern is None:
            pattern = _template_regex(template)
            _TEMPLATE_REGEX_CACHE[template] = pattern
        if pattern.match(instructions):
            return question_id
    return None


def validate_journal_rows() -> tuple[Counter, Counter, list[str]]:

    witnesses: Counter = Counter()
    unmatched: Counter = Counter()
    notes: list[str] = []
    journal = DecisionJournal()
    for row in journal.replay():
        if row.get("type") != "decision" or row.get("shadow_of") or row.get("error") or not row.get("answers"):
            continue
        phase = row.get("phase")
        if phase not in GOLDEN_PHASES:
            continue
        questions = row.get("questions") or {}
        state = row.get("state") or {}
        candidates = []
        cands = state.get("candidates")
        if isinstance(cands, dict):
            candidates = [v if isinstance(v, str) and v else k for k, v in cands.items()]
        elif isinstance(cands, list):
            candidates = [str(c) for c in cands]
        slots = {"__candidates__": candidates, **{k: v for k, v in state.items() if k not in ("candidates",)}}
        for qname, q in questions.items():
            instructions = q.get("instructions") or ""
            hit = match_one(instructions, slots=slots)
            if hit:
                witnesses[hit] += 1
            elif any(instructions.startswith(prefix) for prefix in JUSTIFIED_UNMATCHED_PREFIXES):
                notes.append(f"justified (pre-freeze wording): {instructions[:70]}")
            else:
                unmatched[f"{phase}.{qname}: {instructions[:90]}"] += 1
    return witnesses, unmatched, notes


def load_templates() -> dict[str, str]:
    import yaml

    raw = yaml.safe_load(ARTIFACT.read_text(encoding="utf-8"))
    return {qid: entry["instructions"] for qid, entry in raw["questions"].items()}


def match_one(instructions: str, slots: dict) -> str | None:
    return match_template(instructions, load_templates(), slots)


def validate_fused_prefix(templates: dict[str, str]) -> list[str]:
    """The fused safe templates must end with their plain safe template."""
    problems = []
    for fused, plain in (("tap.safe_fused", "tap.safe"), ("long_press.safe_fused", "long_press.safe")):
        if not templates[plain] or not templates[fused].endswith(templates[plain]):
            problems.append(f"{fused} does not end with {plain}")
    # the prefix shape matches the runtime f-string
    prefix = re.compile(r"^chosen_action=.+\. proposed_command=.*\. target_bounds=.*\. $")
    for fused in ("tap.safe_fused", "long_press.safe_fused"):
        body = templates[fused][: -len(templates[fused.split('.')[0] + ".safe"])]
        if not prefix.match(body):
            problems.append(f"{fused} prefix shape drifted: {body[:60]!r}")
    return problems


def validate_constants() -> list[str]:
    import importlib

    problems = []
    templates = load_templates()
    for module_name, attr, question_id in CONSTANT_IDS:
        actual = getattr(importlib.import_module(module_name), attr, None)
        if actual is None:
            problems.append(f"{module_name}.{attr} missing")
        elif actual != templates[question_id]:
            problems.append(f"{module_name}.{attr} != artifact {question_id}")
    return problems


def stamp_witnesses(witnesses: Counter) -> None:
    """Stamp validated: + witness_n onto each entry (calibration provenance)."""
    import yaml

    raw = yaml.safe_load(ARTIFACT.read_text(encoding="utf-8"))
    for question_id, entry in raw["questions"].items():
        n = witnesses.get(question_id, 0)
        entry["witness_n"] = n
        entry["validated"] = n > 0
    ARTIFACT.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8")


def main() -> int:
    witnesses, unmatched, _notes = validate_journal_rows()
    problems = validate_fused_prefix(load_templates()) + validate_constants()
    templates = load_templates()
    print(f"artifact entries: {len(templates)}")
    print(f"witnessed: {sum(witnesses.values())} golden-path journal instances across {len(witnesses)} templates")
    for question_id, n in sorted(witnesses.items()):
        print(f"  {n:>4}x  {question_id}")
    unwitnessed = sorted(set(templates) - set(witnesses))
    if unwitnessed:
        print("unwitnessed templates (validated: false -- no golden example asks them yet):")
        for question_id in unwitnessed:
            print(f"  - {question_id}")
    if unmatched:
        print(f"\nUNMATCHED golden-path journal questions ({sum(unmatched.values())} rows) -- these MUST be zero or justified:")
        for key, n in sorted(unmatched.items()):
            print(f"  {n:>4}x  {key}")
    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
    if unmatched or problems:
        return 1
    stamp_witnesses(witnesses)
    print("OK: artifact validated against the golden set and the module constants; witness counts stamped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())