"""Question-set tests: the frozen artifact is load-bearing, so it gets the
same guard discipline as the journal's phase labels.

- The artifact loads, is versioned, and every id resolves; templates format.
- The fused safe templates end with their plain safe template.
- Missing slots fail closed (CompiledQuestionError), never silent generation.
- A static guard bans instruction literals outside question_sets (the runtime
  path must consume the artifact, not inline prose).
- The escape hatch works and is journaled: GeneratedNoul's marker fields are
  excluded from the wire dump, ask() journals generated=<source>, compiled
  questions journal generated=None.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from jevdevice import question_sets
from jevdevice.jev import Noul

SRC = Path(__file__).resolve().parent.parent / "src" / "jevdevice"
INSTRUCTION_LITERAL = re.compile(r"(?:Noul|Choice|Score)\(\s*instructions\s*=\s*['\"(]|instructions\s*=\s*['\"]")


def test_artifact_loads_and_version_pins() -> None:
    qset = question_sets.load()
    assert qset.version == question_sets.DEFAULT_VERSION
    assert qset.questions
    for question_id, entry in qset.questions.items():
        assert entry["type"] in ("noul", "choice", "score", "fragment"), question_id
        assert isinstance(entry["instructions"], str) and entry["instructions"], question_id


def test_text_formats_slots_and_fails_closed_on_missing_slots() -> None:
    assert question_sets.text("recall.any") == "Could any of these candidates satisfy the goal?"
    fused = question_sets.text(
        "tap.safe_fused", chosen_action="tap btn", proposed_command="input tap 1 2", target_bounds="[0,0][1,1]",
    )
    assert fused.startswith("chosen_action='tap btn'. proposed_command='input tap 1 2'. target_bounds='[0,0][1,1]'. ")
    # {candidate} is filled later by the fit builder; it passes through here
    assert "{candidate}" in question_sets.text("tap.fit")
    with pytest.raises(question_sets.CompiledQuestionError, match="missing slot"):
        question_sets.text("tap.safe_fused", chosen_action="tap btn")
    with pytest.raises(question_sets.CompiledQuestionError, match="unknown question id"):
        question_sets.text("not.a.real.id")


def test_fused_safe_templates_end_with_the_plain_safe_template() -> None:
    qset = question_sets.load()
    for fused, plain in (("tap.safe_fused", "tap.safe"), ("long_press.safe_fused", "long_press.safe")):
        assert qset.template(fused)["instructions"].endswith(qset.template(plain)["instructions"])


def test_composed_kind_pick_screen_is_the_frozen_sum() -> None:
    qset = question_sets.load()
    assert qset.template("kind.pick_screen")["instructions"] == (
        qset.template("kind.pick")["instructions"] + " " .rstrip() +
        # the suffix (formerly composed at runtime) is byte-frozen inside pick_screen
        qset.template("kind.pick_screen")["instructions"][len(qset.template("kind.pick")["instructions"]):]
    )


def test_fit_questions_build_and_mark_generated_family() -> None:
    compiled = question_sets.fit_questions("tap.fit", ["a", "b"])
    assert list(compiled) == ["fit_0", "fit_1"]
    assert compiled["fit_0"].instructions == "Would tapping a actually perform the goal?"
    assert compiled["fit_0"].model_dump(mode="json", exclude_none=True) == {
        "type": "noul", "instructions": "Would tapping a actually perform the goal?",
    }
    generated = question_sets.fit_questions(
        "tap.fit", ["a"], generated_source="propose_tap.fit_instructions_override",
    )
    assert isinstance(generated["fit_0"], question_sets.GeneratedNoul)
    # the marker never reaches the wire: byte-identical JSON to a plain Noul
    assert generated["fit_0"].model_dump(mode="json", exclude_none=True) == compiled["fit_0"].model_dump(mode="json", exclude_none=True)


def test_generated_noul_wire_invisible() -> None:
    plain = Noul(instructions="does it fit?")
    hatched = question_sets.generated_noul("test.hatch", "does it fit?")
    assert hatched.generated is True and hatched.hatched_from == "test.hatch"
    assert hatched.model_dump(mode="json", exclude_none=True) == plain.model_dump(mode="json", exclude_none=True)
    assert question_sets.generated_source({"q": plain}) is None
    assert question_sets.generated_source({"q": hatched}) == "test.hatch"


class _FakeHTTP:
    """Minimal stand-in for the httpx client: returns one canned payload."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def post(self, url, headers=None, json=None):
        payload = self.payload
        response = type("R", (), {"raise_for_status": lambda self: None, "json": lambda self: payload})()
        return response


async def test_ask_journals_generated_source_and_compiled_stays_none(tmp_path, monkeypatch) -> None:
    from jevdevice.decision_log import DecisionJournal
    from jevdevice.jev import JevClient

    monkeypatch.delenv("JEV_JOURNAL", raising=False)
    journal = DecisionJournal(tmp_path)
    client = JevClient("test-key", journal=journal)
    client._client = _FakeHTTP({"answers": {"q": {"type": "noul", "noul": 0.9}}})
    answers = await client.ask(
        {"goal": "g"},
        {"q": question_sets.generated_noul("propose_tap.fit_instructions_override", "does it fit?")},
        phase="fill",
    )
    assert answers["q"].noul == 0.9
    rows = [r for r in journal.replay() if r.get("type") == "decision"]
    assert rows and rows[-1]["generated"] == "propose_tap.fit_instructions_override"

    # and a plain (compiled-shaped) question journals generated=None
    await client.ask({"goal": "g"}, {"q": Noul(instructions="i")}, phase="fill")
    rows = [r for r in journal.replay() if r.get("type") == "decision"]
    assert rows[-1]["generated"] is None


def test_no_instruction_literals_outside_question_sets() -> None:
    """Static guard: the runtime consumes the frozen set; instruction
    prose lives only in the artifact (eval tooling and tests may inline)."""
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        if path.name == "question_sets" or path.name.startswith("question_sets"):
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if INSTRUCTION_LITERAL.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()[:100]}")
    assert not offenders, "instruction literals must live in the frozen artifact:\n" + "\n".join(offenders)


def test_env_knob_selects_version(monkeypatch) -> None:
    question_sets._cache.clear()
    monkeypatch.setenv(question_sets.ENV_VERSION, "v1")
    assert question_sets.load().version == "v1"
    with pytest.raises(question_sets.CompiledQuestionError):
        question_sets.load("v999")  # unknown artifact -> fail-closed, not a fallback
    question_sets._cache.clear()
    assert json.dumps(question_sets.load("v1").template("recall.any")) is not None
