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

import httpx2
import pytest
import typesymbolic.vocab
from typesymbolic.journal import Journal
from typesymbolic.judge import JevEngine

from jevdevice import question_sets
from jevdevice.jev import Noul, ask
from jevdevice.journal import decision_log
from jevdevice.judge.gate import CommandVariant, finalize_gate

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


def test_unit_maps_picks_fits_gates_and_falls_back_to_none() -> None:
    qset = question_sets.load()
    for qid in ("tap.pick", "kind.pick", "kind.pick_screen", "swipe.pick", "toggle.service",
                "toggle.enabled", "toggle.resolve_status.pick"):
        assert qset.unit(qid) == ("pick", "confidence"), qid
    for qid in ("tap.fit", "toggle.resolve_status.fit", "dumpsys_query.fit"):
        assert qset.unit(qid) == ("fit", "noul_p"), qid
    assert qset.unit("tap.safe") == ("gate", "noul_p")
    assert qset.unit("tap.safe_fused") == ("gate", "noul_p")
    assert qset.unit("gate.safe.default") == ("gate", "noul_p")
    # excluded on purpose: not confidence-gated / accept_any_fitting skips the gate
    for qid in ("type_value.pick", "dumpsys_query.pick", "dumpsys_field.pick", "recall.any"):
        assert qset.unit(qid) is None, qid


def test_tagged_fit_questions_are_byte_identical_and_carry_the_qid() -> None:
    untagged = question_sets.fit_questions("tap.fit", ["a", "b"])
    tagged = question_sets.fit_questions("tap.fit", ["a", "b"], qid="tap.fit")
    assert tagged["fit_0"].instructions == untagged["fit_0"].instructions == "Would tapping a actually perform the goal?"
    assert tagged["fit_0"].model_dump(mode="json", exclude_none=True) == untagged["fit_0"].model_dump(mode="json", exclude_none=True)
    assert isinstance(tagged["fit_0"], question_sets.TaggedNoul)
    assert tagged["fit_0"].qid == "tap.fit"
    assert not isinstance(untagged["fit_0"], question_sets.TaggedNoul)


def test_generated_noul_wire_invisible() -> None:
    plain = Noul(instructions="does it fit?")
    hatched = question_sets.generated_noul("test.hatch", "does it fit?")
    assert hatched.generated is True and hatched.hatched_from == "test.hatch"
    assert hatched.model_dump(mode="json", exclude_none=True) == plain.model_dump(mode="json", exclude_none=True)
    assert question_sets.generated_source({"q": plain}) is None
    assert question_sets.generated_source({"q": hatched}) == "test.hatch"


def _canned_answer(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json={
        "model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 1},
        "answers": {"q": {"type": "noul", "noul": 0.9}},
    })


async def test_ask_journals_generated_source_and_compiled_stays_none(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("JEV_JOURNAL", raising=False)
    journal = Journal(root=tmp_path, background_writes=False)
    monkeypatch.setattr(decision_log, "_default_journal", journal)
    engine = JevEngine(api_key="test-key", transport=httpx2.MockTransport(_canned_answer))
    _, answers = await ask(
        engine,
        {"goal": "g"},
        {"q": question_sets.generated_noul("propose_tap.fit_instructions_override", "does it fit?")},
        phase="fill",
    )
    assert answers["q"].noul == 0.9
    rows = [r for r in journal.replay() if r.get("type") == "decision"]
    assert rows and rows[-1]["extra"]["generated"] == "propose_tap.fit_instructions_override"

    # and a plain (compiled-shaped) question journals no generated entry
    await ask(engine, {"goal": "g"}, {"q": Noul(instructions="i")}, phase="fill")
    rows = [r for r in journal.replay() if r.get("type") == "decision"]
    assert rows[-1].get("extra", {}).get("generated") is None


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


# --- typesymbolic Vocabulary protocol conformance -------------------------

def test_question_set_conforms_to_the_core_vocabulary_protocol() -> None:
    """QuestionSet is a typesymbolic vocab.Vocabulary: `version` + `ask(qid,
    **slots)`, fail-closed exactly like the native helpers."""
    vocab: typesymbolic.vocab.Vocabulary = question_sets.load()  # structural conformance, typed
    assert vocab.version == "v1"

    q = vocab.ask("recall.any")
    assert isinstance(q, question_sets.Noul) and q.instructions

    picked = vocab.ask("tap.pick", criteria={"elem one": None, "elem two": "described"})
    assert isinstance(picked, question_sets.Choice)
    assert picked.criteria == {"elem one": None, "elem two": "described"}


def test_vocabulary_ask_fails_closed_like_the_native_helpers() -> None:
    vocab = question_sets.load()
    with pytest.raises(question_sets.CompiledQuestionError):
        vocab.ask("no.such.question_id")  # unknown id -> fail-closed, not improvised
    with pytest.raises(question_sets.CompiledQuestionError):
        vocab.ask("planner.step_still_fits")  # missing required slots -> fail-closed


def test_core_gate_types_are_the_gate_types() -> None:
    """The gate verdicts/results are typesymbolic's, with this repo's reason
    strings and the raw-noul (not synthesized) confidence."""
    from typesymbolic.gate import GateResult as CoreGateResult
    from typesymbolic.gate import GateVerdict as CoreGateVerdict

    command = CommandVariant("svc bluetooth disable", "matches the goal")
    approved = finalize_gate(command, confidence=0.9, threshold=0.8)
    assert isinstance(approved, CoreGateResult)
    assert approved.verdict == CoreGateVerdict.ACT and approved.reason == "jev_confirmed"
    below = finalize_gate(command, confidence=0.5, threshold=0.8)
    assert below.verdict == CoreGateVerdict.NEEDS_APPROVAL and below.confidence == 0.5
    denied = finalize_gate(CommandVariant("rm -rf /sdcard", "nope"), confidence=0.99)
    assert denied.verdict == CoreGateVerdict.DENY and denied.reason == "deny_listed"
    missing = finalize_gate(command, confidence=None)  # type: ignore[arg-type]
    assert missing.verdict == CoreGateVerdict.NEEDS_APPROVAL  # fails closed, unlike core's mutation_gate
