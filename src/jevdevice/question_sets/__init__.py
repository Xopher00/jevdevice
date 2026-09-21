"""Question sets: versioned, frozen compile-time artifacts of every question
phrasing the runtime can ask.

The phrasings are compiled artifacts under question_sets/v{N}.yaml: the
journal is the labeled-example source the compile step validates against
(eval/phases/compile_questions.py), and the runtime CONSUMES the frozen set
-- it never invents phrasings at runtime.

Two load-bearing rules:

- FAIL-CLOSED: an unknown question_id or a missing format slot raises
  CompiledQuestionError -- a typo in an id can never silently become a
  runtime-generated question.
- RUNTIME GENERATION IS THE ESCALATION ESCAPE HATCH ONLY: a question built
  outside the frozen set (e.g. a multi-step caller overriding fit wording)
  must be built via generated_noul()/generated_choice() and is journaled
  (decision row `generated=<source>`) so it can be promoted into the next
  compiled set. No golden-path call site uses it (static guard test).

The wire contract is untouched: GeneratedNoul's marker fields are
`Field(exclude=True)`, so `model_dump()` produces byte-identical JSON --
the ask() wire body stays exactly {model, state, questions}.
"""

from __future__ import annotations

import os
import threading
from importlib import resources

import yaml
from pydantic import Field

from jevdevice.jev import Choice, Noul, Question, Score

DEFAULT_VERSION = "v1"
ENV_VERSION = "JEV_QUESTION_SET"  # named knob: select which frozen set the runtime consumes


class CompiledQuestionError(RuntimeError):
    """Raised when a question id is unknown, a required slot is missing, or the
    frozen artifact itself fails validation. Fail-closed: the runtime never
    falls back to improvising a question."""


class GeneratedNoul(Noul):
    """A runtime-generated (escape-hatch) Noul. Carries its provenance on
    excluded fields so the wire body is byte-identical to a plain Noul, but
    ask() can see it and journal `generated=<source>`."""

    generated: bool = Field(default=True, exclude=True)
    hatched_from: str = Field(default="", exclude=True)


def generated_noul(source: str, instructions: str, criteria: dict[str, str] | None = None) -> GeneratedNoul:
    """The escape hatch: build a question OUTSIDE the frozen set. The decision
    row journals `generated=<source>`; generated questions are eligible for
    promotion into the next compiled set. Never use on the normal runtime path."""
    return GeneratedNoul(instructions=instructions, criteria=criteria, hatched_from=source)


def generated_source(questions: dict) -> str | None:
    """ask() calls this: the hatched_from of the first generated question, or
    None when every question came from the frozen set (the normal case)."""
    for question in questions.values():
        if isinstance(question, GeneratedNoul):
            return question.hatched_from
        if getattr(question, "generated", False):
            return str(getattr(question, "hatched_from", "") or "unknown")
    return None


class _KeepUnknownSlots(dict):
    """format_map helper: known slots fill in, unknown ones (e.g. {candidate},
    filled later per-candidate by fit_questions) pass through untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class QuestionSet:
    """One frozen question-set artifact. Criteria (option maps, option
    descriptions) are runtime data and stay at the call sites -- the artifact
    freezes PHRASINGS, exactly what was mutable prose before."""

    def __init__(self, version: str, data: dict) -> None:
        self.version = version
        self.questions: dict[str, dict] = data.get("questions") or {}
        if not self.questions:
            raise CompiledQuestionError(f"question set {version} is empty")
        for question_id, entry in self.questions.items():
            if entry.get("type") not in ("noul", "choice", "score"):
                raise CompiledQuestionError(f"{question_id}: bad type {entry.get('type')!r}")
            if not isinstance(entry.get("instructions"), str) or not entry["instructions"]:
                raise CompiledQuestionError(f"{question_id}: missing instructions")

    def template(self, question_id: str) -> dict:
        entry = self.questions.get(question_id)
        if entry is None:
            known = ", ".join(sorted(self.questions))
            raise CompiledQuestionError(f"unknown question id {question_id!r} in question set {self.version} (have: {known})")
        return entry

    def _slots(self, question_id: str, slots: dict) -> str:
        template = self.template(question_id)["instructions"]
        required = {
            name.split("!")[0].split(":")[0]
            for placeholder in _placeholders(template)
            if (name := placeholder) != "candidate"
        }
        missing = required - slots.keys()
        if missing:
            raise CompiledQuestionError(f"{question_id}: missing slot(s) {sorted(missing)} -- fail-closed, not a silent generation")
        return template.format_map(_KeepUnknownSlots(slots))

    def text(self, question_id: str, **slots: object) -> str:
        """The formatted instructions string -- for call sites that thread
        phrasing into shared builders (narrowing's fit/pick parameters)."""
        return self._slots(question_id, slots)

    def noul(self, question_id: str, criteria: dict | None = None, **slots: object) -> Noul:
        return Noul(instructions=self.text(question_id, **slots), criteria=criteria)

    def choice(self, question_id: str, criteria: dict[str, str | None], **slots: object) -> Choice:
        return Choice(instructions=self.text(question_id, **slots), criteria=criteria)

    def score(self, question_id: str, criteria: list[str], **slots: object) -> Score:
        return Score(instructions=self.text(question_id, **slots), criteria=criteria)

    def fit_questions(
        self, question_id: str, candidates, describe=lambda c: c, *, generated_source: str | None = None,
    ) -> dict[str, Question]:
        """The fit_{i} Noul family (narrowing.fit_questions' phrasing source).
        generated_source != None marks the WHOLE family as an escape-hatch
        generation (a caller overrode the wording) -- journaled, promotable."""
        return fit_questions_from(self.text(question_id), candidates, describe, generated_source=generated_source)


_cache: dict[str, QuestionSet] = {}
_lock = threading.Lock()


def load(version: str | None = None) -> QuestionSet:
    """Load + cache one frozen question set. JEV_QUESTION_SET selects which
    version the runtime consumes (default v1) -- shadow-running a v2 against
    the same code is a knob, not a code change."""
    version = version or os.environ.get(ENV_VERSION) or DEFAULT_VERSION
    with _lock:
        if version not in _cache:
            package = resources.files("jevdevice.question_sets")
            path = package / f"{version}.yaml"
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            except FileNotFoundError as missing:
                raise CompiledQuestionError(f"unknown question set {version!r}") from missing
            if raw.get("version") != version:
                raise CompiledQuestionError(f"artifact version {raw.get('version')!r} does not match filename {version!r}")
            _cache[version] = QuestionSet(version, raw)
    return _cache[version]


# module-level conveniences so call sites read like prose:
#   questions = {"any": question_sets.noul("recall.any"), ...}
def text(question_id: str, **slots: object) -> str:
    return load().text(question_id, **slots)


def noul(question_id: str, criteria: dict | None = None, **slots: object) -> Noul:
    return load().noul(question_id, criteria=criteria, **slots)


def choice(question_id: str, criteria: dict[str, str | None], **slots: object) -> Choice:
    return load().choice(question_id, criteria=criteria, **slots)


def fit_questions(question_id: str, candidates, describe=lambda c: c, *, generated_source: str | None = None) -> dict[str, Question]:
    return load().fit_questions(question_id, candidates, describe, generated_source=generated_source)


def _placeholders(template: str) -> list[str]:
    import string

    return [name for _, name, _, _ in string.Formatter().parse(template) if name]


def fit_questions_from(instructions: str, candidates, describe=lambda c: c, *, generated_source: str | None = None) -> dict[str, Question]:
    """Shared fit_{i} builder: one "does this candidate really fit" Noul per
    candidate, keyed fit_0..n in candidate order. generated_source != None
    marks the family as runtime-generated (escape hatch), which ask() journals."""
    out: dict[str, Question] = {}
    for i, candidate in enumerate(candidates):
        formatted = instructions.format(candidate=describe(candidate))
        out[f"fit_{i}"] = (
            GeneratedNoul(instructions=formatted, hatched_from=generated_source)
            if generated_source
            else Noul(instructions=formatted)
        )
    return out


def fit_questions_from_text(question_id: str, candidates, describe=lambda c: c, *, generated_source: str | None = None) -> dict[str, Question]:
    return fit_questions_from(text(question_id), candidates, describe, generated_source=generated_source)