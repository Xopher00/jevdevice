"""Act on the real, currently on-screen UI (see elements.py for how it's parsed): narrow real
elements + gate the resulting tap/long-press/type/swipe, following the same
"real enumeration -> semantic narrow -> deterministic action -> Jev verify" pattern app_launch.py
uses for packages and services.py uses for dumpsys services.
"""

from __future__ import annotations

import asyncio
import shlex
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from jevdevice import question_sets
from jevdevice.budget import choice_criteria, current_profile, is_abstain
from jevdevice.device import Device
from jevdevice.jev import Choice, JevClient, Noul
from jevdevice.judge.gate import (
    ClosedSetProposal,
    CommandVariant,
    GateResult,
    Pending,
    finalize_gate,
    gate_command,
    propose_from_closed_set,
    resolve_gate,
)
from jevdevice.judge.narrowing import (
    _abstain_verdict,
    extract_fits,
    fit_questions,
    narrow_and_pick,
)
from jevdevice.matching import NarrowVerdict, decide, extract_value_spans, fuzzy_narrow

from .elements import (
    Element,
    describe_screen,
    dump_screen,
    foreground_package,
    parse_actionable_elements,
    parse_all_elements,
    parse_editable_elements,
    parse_long_clickable_elements,
    short_options,
)
from .services import execute_command

# These module constants are SOURCED from the
# versioned artifact (question_sets/v1.yaml) so calibrate/ CLIs and tests that
# import them keep working -- the artifact is the single source of truth.
TAP_SAFE_INSTRUCTIONS = question_sets.text("tap.safe")

TYPE_SAFE_INSTRUCTIONS = question_sets.text("type.safe")


class _LRUCache(OrderedDict):
    """Bounded dict: evicts the least-recently-used entry once over `maxsize`."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        if key not in self:
            return default
        return self[key]

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.maxsize:
            self.popitem(last=False)


# (package, goal) -> label; only used when that label is still present in the current real dump.
_ELEMENT_CACHE: _LRUCache = _LRUCache(maxsize=500)


@dataclass
class ActionOutcome:
    element: str | None
    confidence: float
    tapped: bool
    satisfied: float
    reasons: tuple[str, ...] = ()


@dataclass
class TapProposal:
    """Result of narrowing + gating a tap goal, before any real tap runs.
    `pending` is set only when the gate needs a decision; `ready` is set when
    it's already approved (or read-only) and can be executed immediately."""
    element: str | None
    confidence: float
    fit: float
    ready: CommandVariant | None = None
    pending: Pending | None = None
    reasons: tuple[str, ...] = ()
    gate_result: GateResult | None = None  # journal linkage: outcome rows read gate_result.call_id


async def _fused_pick_and_gate(
    jev: JevClient, goal: str, options: dict[str, Element], *,
    pick_instructions: str, fit_instructions: str, safe_fused_id: str, safe_instructions: str,
    command_for, chosen_label_for, fit_generated_source: str | None = None,
) -> tuple[NarrowVerdict, GateResult | None]:
    """Narrow + gate in one batched ask instead of two sequential round trips: a per-candidate
    safety Noul (TypeSafe's speculative-fan-out pattern) is asked alongside the pick, using each
    candidate's own real bounds/command, so the winner's gate answer is already in hand. Shared by
    tap and long_press -- only the command/label shape differs between them. The per-candidate
    safety wording is the frozen fused template (safe_fused_id, e.g. tap.safe_fused); the plain
    safe_instructions ride along for the non-fused gate fallback below.

    `options` is the option map the judge answers over (short raw labels on profiles with
    short_labels, full labels otherwise); option keys double as the dict keys for the
    winner's coordinates/bounds."""
    profile = current_profile(jev.engine_name)
    if profile.descriptions_in_state:
        # Rich descriptions ride in the state body; the option list in the head stays short.
        criteria = {c: (element.description or None) for c, element in options.items()}
    else:
        criteria = {c: None for c in options}
    safe_keys = {f"safe_{i}": c for i, c in enumerate(options)}
    questions = {
        "pick": Choice(instructions=pick_instructions, criteria=choice_criteria(criteria, profile)),
        **fit_questions(options, fit_instructions, lambda c: options[c].description or c, generated_source=fit_generated_source),
        **{
            key: Noul(instructions=question_sets.text(
                safe_fused_id,
                chosen_action=chosen_label_for(c),
                proposed_command=command_for(options[c]),
                target_bounds=options[c].bounds,
            ))
            for key, c in safe_keys.items()
        },
    }
    # One call_id covers the fused pick + its own gate answer, so the downstream
    # outcome row joins the single ask that produced both.
    call_id = str(uuid.uuid4())
    answers = await jev.ask({"goal": goal, "candidates": criteria}, questions, phase="recall", call_id=call_id)
    pick = answers["pick"]
    fits = extract_fits(answers, options)
    if is_abstain(pick.choice):
        return _abstain_verdict(options, fits, pick.confidence), None
    verdict = decide(pick.choice, pick.probabilities, pick.confidence, fits, list(options),
                     min_fit=profile.min_fit, min_confidence=profile.min_confidence, min_margin=profile.min_margin)
    if not verdict.ok:
        return verdict, None
    command = CommandVariant(command=command_for(options[verdict.choice]), rationale=f"{chosen_label_for(verdict.choice)} per the goal")
    safe_by_candidate = {c: answers[key].noul for key, c in safe_keys.items()}
    return verdict, finalize_gate(command, safe_by_candidate[verdict.choice], profile.gate_threshold, call_id=call_id)


async def _cached_or_narrow(
    cache_key: tuple[str | None, str], elements: dict[str, Element],
    narrow: Callable[[], Awaitable[NarrowVerdict]], *, verbose: bool = False,
) -> NarrowVerdict:
    """Reuse the cached label if it's still on screen (skipping `narrow` entirely);
    otherwise run it and cache its pick on success."""
    cached = _ELEMENT_CACHE.get(cache_key)
    if cached in elements:
        if verbose:
            print(f"cache hit: {cached!r} still on screen, skipping Jev narrowing")
        return NarrowVerdict(cached, 1.0, 1.0, 1.0, [cached])
    verdict = await narrow()
    if verdict.ok:
        _ELEMENT_CACHE[cache_key] = verdict.choice
    return verdict


async def _propose_gesture(
    jev: JevClient, device: Device, goal: str, *,
    parse_elements, pick_instructions: str, fit_instructions: str, safe_fused_id: str,
    safe_question_id: str, command_for, chosen_label_for, verbose: bool = True,
    fit_generated_source: str | None = None,
) -> TapProposal:
    """Shared by tap and long_press: narrow real matching elements + gate the resulting
    gesture. Only the element filter and command/label shape differ between callers.
    safe_fused_id/safe_question_id are frozen question-set ids (tap vs long_press).
    fit_generated_source != None marks the fit family as an escape-hatch generation."""
    safe_instructions = question_sets.text(safe_question_id)
    dump_xml = await dump_screen(device)
    elements = parse_elements(dump_xml)
    if verbose:
        print(f"{len(elements)} real matching elements on screen")

    profile = current_profile(jev.engine_name)
    # The option map the judge answers over: short raw labels where the profile
    # asks for them, the full labels otherwise. Everything downstream (cache,
    # coordinates, gate evidence) keys off this map, so the judge's answer
    # always resolves straight back to the real element.
    options = short_options(elements) if profile.short_labels else elements
    cache_key = (foreground_package(dump_xml), goal)
    gate_result: GateResult | None = None

    async def _narrow() -> NarrowVerdict:
        nonlocal gate_result
        if options and len(options) <= profile.chunk_size:
            verdict, gate_result = await _fused_pick_and_gate(
                jev, goal, options,
                pick_instructions=pick_instructions, fit_instructions=fit_instructions,
                safe_fused_id=safe_fused_id, safe_instructions=safe_instructions,
                command_for=command_for, chosen_label_for=chosen_label_for,
                fit_generated_source=fit_generated_source,
            )
            if verbose:
                print(f"Jev picked: {verdict.choice} (confidence {verdict.confidence:.2f}, fit {verdict.fit:.2f})")
        else:
            verdict = await narrow_and_pick(jev, goal, list(options), instructions=pick_instructions, fit_instructions=fit_instructions, describe=lambda c: options[c].description or c, fit_generated_source=fit_generated_source)
            if verbose:
                print(f"shortlist: {verdict.shortlist}")
                print(f"Jev picked: {verdict.choice} (confidence {verdict.confidence:.2f}, fit {verdict.fit:.2f})")
        return verdict

    verdict = await _cached_or_narrow(cache_key, options, _narrow, verbose=verbose)

    if not verdict.ok:
        if verbose:
            print(f"=== ESCALATED === {'; '.join(verdict.reasons)}")
        return TapProposal(None, verdict.confidence, verdict.fit, reasons=tuple(verdict.reasons))

    target = options[verdict.choice]
    chosen_label = chosen_label_for(verdict.choice)
    command = CommandVariant(command=command_for(target), rationale=f"{chosen_label} per the goal")
    if gate_result is None:
        gate_result = await gate_command(
            jev, command, chosen_label=chosen_label,
            evidence={"target_element": verdict.choice, "target_bounds": target.bounds},
            instructions=question_sets.text(safe_question_id),
        )
    if verbose:
        print(f"gate verdict: {gate_result.verdict} ({gate_result.reason}, noul={gate_result.confidence})")
    ready, pending, reasons = resolve_gate(gate_result, command, chosen_label)
    return TapProposal(verdict.choice, verdict.confidence, verdict.fit, ready, pending, reasons, gate_result=gate_result)


async def propose_tap(
    jev: JevClient, device: Device, goal: str, *, verbose: bool = True,
    fit_instructions: str | None = None,
) -> TapProposal:
    """Narrow real on-screen elements + gate the resulting tap. No execution. The fit wording
    comes from the frozen question set (tap.fit). A multi-step caller (see
    experiment/action_chain.py) can override it to ask about progress instead -- an override is
    a RUNTIME-GENERATED question (the escalation escape hatch): it is journaled with
    generated=<source> and is eligible for promotion into the next compiled question set."""
    return await _propose_gesture(
        jev, device, goal,
        parse_elements=parse_actionable_elements,
        pick_instructions=question_sets.text("tap.pick"),
        fit_instructions=fit_instructions or question_sets.text("tap.fit"),
        fit_generated_source=None if fit_instructions is None else "propose_tap.fit_instructions_override",
        safe_fused_id="tap.safe_fused",
        safe_question_id="tap.safe",
        command_for=lambda el: f"input tap {el.x} {el.y}",
        chosen_label_for=lambda c: f"tap {c}",
        verbose=verbose,
    )


LONG_PRESS_SAFE_INSTRUCTIONS = question_sets.text("long_press.safe")


async def propose_long_press(
    jev: JevClient, device: Device, goal: str, *, verbose: bool = True,
    fit_instructions: str | None = None,
) -> TapProposal:
    """Narrow real long-clickable elements + gate the resulting long-press. No execution --
    execute_tap runs it (a long-press is just `input swipe` with start==end). Same escape-hatch
    rule as propose_tap: an explicit fit_instructions override is runtime-generated and journaled."""
    return await _propose_gesture(
        jev, device, goal,
        parse_elements=parse_long_clickable_elements,
        pick_instructions=question_sets.text("long_press.pick"),
        fit_instructions=fit_instructions or question_sets.text("long_press.fit"),
        fit_generated_source=None if fit_instructions is None else "propose_long_press.fit_instructions_override",
        safe_fused_id="long_press.safe_fused",
        safe_question_id="long_press.safe",
        command_for=lambda el: f"input swipe {el.x} {el.y} {el.x} {el.y} 800",
        chosen_label_for=lambda c: f"long-press {c}",
        verbose=verbose,
    )


# The four real physical swipe directions a touchscreen has -- like DND_MODES/TOGGLEABLE_SERVICES,
# a genuinely fixed small set, not an app-specific menu of hardcoded options.
DIRECTIONS = {"up": None, "down": None, "left": None, "right": None}

SWIPE_SAFE_INSTRUCTIONS = question_sets.text("swipe.safe")


async def propose_swipe(jev: JevClient, device: Device, goal: str, *, verbose: bool = True) -> ClosedSetProposal:
    """Pick one of the four real swipe directions for the goal + gate it. No execution."""
    width, height = await device.window_size()
    cx, cy = width // 2, height // 2
    dx, dy = int(width * 0.3), int(height * 0.3)
    # A "down" scroll (reveal content further down) is a physical swipe upward: finger starts
    # low on screen (larger y) and moves high (smaller y); "left"/"right" mirror this on the x axis.
    endpoints = {
        "up": ((cx, cy - dy), (cx, cy + dy)),
        "down": ((cx, cy + dy), (cx, cy - dy)),
        "left": ((cx + dx, cy), (cx - dx, cy)),
        "right": ((cx - dx, cy), (cx + dx, cy)),
    }

    def command_for(direction: str) -> CommandVariant:
        (x1, y1), (x2, y2) = endpoints[direction]
        return CommandVariant(command=f"input swipe {x1} {y1} {x2} {y2} 300", rationale=f"swipe {direction} per the goal")

    return await propose_from_closed_set(
        jev, goal, DIRECTIONS,
        options_key="direction_options",
        pick_instructions=question_sets.text("swipe.pick"),
        any_fit_instructions=question_sets.text("swipe.any_fit"),
        pick_verb="direction",
        command_for=command_for,
        label_for=lambda d: f"swipe {d}",
        gate_instructions=SWIPE_SAFE_INSTRUCTIONS,
        verbose=verbose,
    )


@dataclass
class ScrollToFindOutcome:
    found: str | None
    attempts: int
    reasons: tuple[str, ...] = ()
    # Journal linkage + what actually ran: the screen-check verdict that resolved
    # (or the last one when it never did), and every swipe command executed on the way.
    call_id: str | None = None
    executed: tuple[str, ...] = ()


async def scroll_to_find(
    jev: JevClient, device: Device, goal: str, *,
    direction: str = "down", max_attempts: int = 8, verbose: bool = True,
) -> ScrollToFindOutcome:
    """Re-checks the real screen against the goal every attempt and swipes only when the
    target genuinely isn't there yet -- reuses narrow_and_pick (is it visible now?) and
    propose_swipe/execute_swipe (one real gesture) instead of paging a fixed number of times."""
    executed: list[str] = []
    last_call_id: str | None = None
    for attempt in range(1, max_attempts + 1):
        elements = parse_all_elements(await dump_screen(device))
        options = short_options(elements) if current_profile(jev.engine_name).short_labels else elements
        verdict = await narrow_and_pick(
            jev, goal, list(options),
            instructions=question_sets.text("scroll_to_find.pick"),
            fit_instructions=question_sets.text("scroll_to_find.fit"),
            describe=lambda c, options=options: options[c].description or c,  # bind now: B023 (lambda is consumed within this iteration)
        )
        last_call_id = verdict.call_id
        if verbose:
            print(f"attempt {attempt}/{max_attempts}: picked {verdict.choice!r} ok={verdict.ok}")
        if verdict.ok:
            return ScrollToFindOutcome(verdict.choice, attempt, call_id=verdict.call_id, executed=tuple(executed))

        proposal = await propose_swipe(jev, device, f"scroll {direction}", verbose=False)
        # Only proposal.ready runs here -- a needs_approval verdict stops the loop like
        # any other unapproved command, it is never executed implicitly.
        if proposal.ready is None:
            return ScrollToFindOutcome(
                None, attempt, reasons=("swipe gate did not approve scrolling", *proposal.reasons),
                call_id=last_call_id, executed=tuple(executed),
            )
        executed.append(proposal.ready.command)
        await execute_command(device, proposal.ready)
    return ScrollToFindOutcome(None, max_attempts, reasons=(f"not found after {max_attempts} scrolls"), call_id=last_call_id, executed=tuple(executed))


async def _verify_after_action(
    jev: JevClient, device: Device, goal: str, acted_on: str, *,
    delays: tuple[float, ...], verbose: bool,
) -> float:
    """Shared by execute_tap/execute_type: re-probe the real screen and ask
    Jev whether the goal is now achieved, retrying with backoff."""
    satisfied = 0.0
    for delay in delays:
        await asyncio.sleep(delay)
        # A fixed-length raw-XML truncation can cut off the real evidence entirely (confirmed
        # live); compact per-element labels carry far more real signal per character.
        truncation: dict = {}
        screen_after = describe_screen(await dump_screen(device), goal=goal, limit=current_profile(jev.engine_name).screen_limit, telemetry=truncation)
        answers = await jev.ask(
            {"goal": goal, "acted_on": acted_on, "screen_after": screen_after},
            {"satisfied": question_sets.noul("verify.satisfied_after_action")},
            phase="verify", truncation=truncation,
        )
        satisfied = answers["satisfied"].noul
        if satisfied >= current_profile(jev.engine_name).noul_floor:
            break
    if verbose:
        print(f"goal met: {satisfied >= 0.5} (noul={satisfied:.2f})")
    return satisfied


async def _execute(
    jev: JevClient, device: Device, goal: str, element: str, command: CommandVariant, *,
    confidence: float, verify: bool, delays: tuple[float, ...], verbose: bool,
) -> ActionOutcome:
    """Shared by execute_tap/execute_type: run the approved command, then verify
    unless the caller opts out (see verify=False docstring on either wrapper)."""
    await device.run(command.command)
    if not verify:
        return ActionOutcome(element, confidence, True, 0.0, reasons=("verify skipped",))
    satisfied = await _verify_after_action(jev, device, goal, element, delays=delays, verbose=verbose)
    return ActionOutcome(element, confidence, True, satisfied)


async def execute_tap(
    jev: JevClient, device: Device, goal: str, element: str, command: CommandVariant, *,
    confidence: float = 0.0, verify: bool = True, delays: tuple[float, ...] = (0.0, 0.0, 0.0), verbose: bool = True,
) -> ActionOutcome:
    """Run an approved tap. verify=False skips the post-tap dump+Jev-ask
    (~2.9s) for a caller that will check the resulting state itself -- e.g.
    an agent sequencing several steps before its own screenshot/inspection."""
    return await _execute(jev, device, goal, element, command, confidence=confidence, verify=verify, delays=delays, verbose=verbose)


@dataclass
class TypeProposal:
    """Result of narrowing a target field + extracting/picking a value + gating,
    before any real typing runs. Same ready/pending/reasons shape as TapProposal."""
    element: str | None
    value: str | None
    confidence: float
    ready: CommandVariant | None = None
    pending: Pending | None = None
    reasons: tuple[str, ...] = ()
    gate_result: GateResult | None = None  # journal linkage: outcome rows read gate_result.call_id


def _no_editable_field_reasons(dump_xml: str, goal: str) -> tuple[str, ...]:
    """A facade search bar (e.g. a real button styled to look like a field) is a real,
    common case: zero real EditText/AutoCompleteTextView nodes, but a real clickable
    candidate that would open one. Name it so the calling agent knows to try a tap first --
    device_do still never chains actions itself."""
    reasons = ["no real editable fields (EditText/AutoCompleteTextView) on screen"]
    clickable = parse_actionable_elements(dump_xml)
    if clickable:
        nearest = fuzzy_narrow(goal, list(clickable), limit=3)
        reasons.append(
            f"{len(clickable)} real clickable elements are present and one may need to be "
            f"tapped first to open a real field, e.g. {', '.join(nearest)}"
        )
    return tuple(reasons)


async def _pick_value(jev: JevClient, goal: str, spans: list[str]) -> dict | None:
    if not spans:
        return None
    profile = current_profile(jev.engine_name)
    return await jev.ask(
        {"goal": goal, "candidate_values": spans},
        {
            "value": question_sets.choice("type_value.pick", choice_criteria({s: None for s in spans}, profile)),
            "any_fit": question_sets.noul("type_value.any_fit"),
        },
        phase="fill",
    )


async def _fused_field_and_value(
    jev: JevClient, goal: str, options: dict[str, Element], spans: list[str], fit_instructions: str,
    *, fit_generated_source: str | None = None,
) -> tuple[NarrowVerdict, dict | None]:
    """Batches field-narrow with value-pick into one real request on a cache miss -- they're
    independent facts that previously cost two separate round trips (asyncio.gather only
    overlaps wall-clock time, it doesn't merge the payloads). `options` is the judge's
    option map (short labels on profiles with short_labels, full labels otherwise)."""
    profile = current_profile(jev.engine_name)
    if profile.descriptions_in_state:
        field_criteria = {c: (element.description or None) for c, element in options.items()}
    else:
        field_criteria = {c: None for c in options}
    state = {"goal": goal, "candidates": field_criteria}
    questions = {
        "pick": question_sets.choice("type_field.pick", choice_criteria(field_criteria, profile)),
        **fit_questions(options, fit_instructions, lambda c: options[c].description or c, generated_source=fit_generated_source),
    }
    if spans:
        state["candidate_values"] = spans
        questions["value"] = question_sets.choice("type_value.pick", choice_criteria({s: None for s in spans}, profile))
        questions["any_fit_value"] = question_sets.noul("type_value.any_fit")
    answers = await jev.ask(state, questions, phase="fill")
    pick = answers["pick"]
    fits = extract_fits(answers, options)
    if is_abstain(pick.choice):
        return _abstain_verdict(options, fits, pick.confidence), None
    # A field's label embeds its live text, so it reads as a new candidate once typed into --
    # same fix as run_dumpsys_query's answer-field pick: min_fit is the real bar on a small pool.
    loose = {"min_confidence": 0.0, "min_margin": 0.0} if len(options) <= 3 else {
        "min_fit": profile.min_fit, "min_confidence": profile.min_confidence, "min_margin": profile.min_margin}
    field_verdict = decide(pick.choice, pick.probabilities, pick.confidence, fits, list(options), **loose)
    value_answers = {"value": answers["value"], "any_fit": answers["any_fit_value"]} if spans else None
    return field_verdict, value_answers


async def propose_type(
    jev: JevClient, device: Device, goal: str, *, verbose: bool = True,
    fit_instructions: str | None = None,
) -> TypeProposal:
    """Narrow real editable fields + extract/pick the real value to type + gate. No execution.
    The fit wording comes from the frozen question set (type_field.fit). An explicit override
    is the escalation escape hatch: runtime-generated and journaled (see propose_tap)."""
    dump_xml = await dump_screen(device)
    elements = parse_editable_elements(dump_xml)
    if verbose:
        print(f"{len(elements)} real editable elements on screen")
    if not elements:
        return TypeProposal(None, None, 0.0, reasons=_no_editable_field_reasons(dump_xml, goal))

    profile = current_profile(jev.engine_name)
    options = short_options(elements) if profile.short_labels else elements
    spans = extract_value_spans(goal)
    cache_key = (foreground_package(dump_xml), goal)
    cached = _ELEMENT_CACHE.get(cache_key)
    if cached in options:
        if verbose:
            print(f"cache hit: {cached!r} still on screen, skipping Jev narrowing")
        field_verdict = NarrowVerdict(cached, 1.0, 1.0, 1.0, [cached])
        answers = await _pick_value(jev, goal, spans)
    else:
        field_verdict, answers = await _fused_field_and_value(
            jev, goal, options, spans,
            fit_instructions or question_sets.text("type_field.fit"),
            fit_generated_source=None if fit_instructions is None else "propose_type.fit_instructions_override",
        )
        if field_verdict.ok:
            _ELEMENT_CACHE[cache_key] = field_verdict.choice
    if verbose:
        print(f"field shortlist: {field_verdict.shortlist}")
        print(f"Jev picked field: {field_verdict.choice} (confidence {field_verdict.confidence:.2f}, fit {field_verdict.fit:.2f})")
    if not field_verdict.ok:
        if verbose:
            print(f"=== ESCALATED === {'; '.join(field_verdict.reasons)}")
        return TypeProposal(None, None, field_verdict.confidence, reasons=tuple(field_verdict.reasons))
    if answers is None:
        return TypeProposal(field_verdict.choice, None, field_verdict.confidence, reasons=("no candidate text value found in the goal",))
    if verbose:
        print(f"value picked: {answers['value'].choice!r} (any_fit {answers['any_fit'].noul:.2f})")
    if answers["any_fit"].noul < profile.noul_floor:
        return TypeProposal(field_verdict.choice, None, field_verdict.confidence, reasons=("no candidate value fits a text field for this goal",))
    if is_abstain(answers["value"].choice):
        return TypeProposal(field_verdict.choice, None, field_verdict.confidence, reasons=("judge abstained: picked none_of_these for the value to type",))
    value = answers["value"].choice

    target = options[field_verdict.choice]
    # Fixed generous count is safer than len(target.text) (real content can outrun displayed
    # text). One `input keyevent` call takes many keycodes -- avoids 200 separate process spawns.
    clear = f"input keyevent 123{' 67' * 200} && " if target.text else ""
    command = CommandVariant(
        command=f"input tap {target.x} {target.y} && {clear}input text {shlex.quote(value)}",
        rationale=f"type {value!r} into {field_verdict.choice!r} per the goal",
    )
    chosen_label = f"type {value!r} into {field_verdict.choice}"
    gate_result = await gate_command(
        jev, command, chosen_label=chosen_label,
        evidence={"target_element": field_verdict.choice, "target_bounds": target.bounds, "typed_value": value},
        instructions=TYPE_SAFE_INSTRUCTIONS,
    )
    if verbose:
        print(f"gate verdict: {gate_result.verdict} ({gate_result.reason}, noul={gate_result.confidence})")
    ready, pending, reasons = resolve_gate(gate_result, command, chosen_label)
    return TypeProposal(field_verdict.choice, value, field_verdict.confidence, ready, pending, reasons, gate_result=gate_result)


async def execute_type(
    jev: JevClient, device: Device, goal: str, element: str, command: CommandVariant, *,
    confidence: float = 0.0, verify: bool = True, delays: tuple[float, ...] = (0.0, 0.0, 0.0), verbose: bool = True,
) -> ActionOutcome:
    """Run an approved type-into-field action. verify=False skips the
    post-action dump+Jev-ask, same tradeoff as execute_tap."""
    return await _execute(jev, device, goal, element, command, confidence=confidence, verify=verify, delays=delays, verbose=verbose)
