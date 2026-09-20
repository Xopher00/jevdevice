"""Act on the real, currently on-screen UI (see elements.py for how it's parsed): narrow real
elements + gate the resulting tap/long-press/type/swipe, following the same
"real enumeration -> semantic narrow -> deterministic action -> Jev verify" pattern app_launch.py
uses for packages and services.py uses for dumpsys services.
"""

from __future__ import annotations

import asyncio
import shlex
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .elements import (
    Element,
    describe_screen,
    dump_screen,
    foreground_package,
    parse_actionable_elements,
    parse_all_elements,
    parse_editable_elements,
    parse_long_clickable_elements,
)
from .gate import (
    ClosedSetProposal,
    CommandVariant,
    GateResult,
    Pending,
    finalize_gate,
    gate_command,
    propose_from_closed_set,
    resolve_gate,
)
from .jev import Choice, JevClient, Noul
from .matching import NarrowVerdict, decide, extract_value_spans, fuzzy_narrow
from .narrowing import CHUNK_SIZE, extract_fits, fit_questions, narrow_and_pick
from .services import execute_command
from .transport import AdbTransport

TAP_SAFE_INSTRUCTIONS = (
    "Does the proposed_command's target and effect match the chosen_action "
    "(same on-screen element, same tap location -- check target_bounds against the coordinates "
    "in proposed_command)? Answer no if it targets a different element, produces a different "
    "effect, or does anything beyond the chosen_action."
)

TYPE_SAFE_INSTRUCTIONS = (
    "Does the proposed_command's target field and typed value match the chosen_action "
    "(same on-screen field -- check target_bounds against the tap coordinates -- and the same "
    "value)? Answer no if it targets a different field, types a different value, or does anything "
    "beyond the chosen_action."
)


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


async def _fused_pick_and_gate(
    jev: JevClient, goal: str, elements: dict[str, Element], *,
    pick_instructions: str, fit_instructions: str, safe_instructions: str,
    command_for, chosen_label_for,
) -> tuple[NarrowVerdict, GateResult | None]:
    """Narrow + gate in one batched ask instead of two sequential round trips: a per-candidate
    safety Noul (TypeSafe's speculative-fan-out pattern) is asked alongside the pick, using each
    candidate's own real bounds/command, so the winner's gate answer is already in hand. Shared by
    tap and long_press -- only the command/label shape differs between them."""
    labels = list(elements)
    criteria = {c: None for c in labels}
    safe_keys = {f"safe_{i}": c for i, c in enumerate(labels)}
    questions = {
        "pick": Choice(instructions=pick_instructions, criteria=criteria),
        **fit_questions(labels, fit_instructions, lambda c: elements[c].description or c),
        **{
            key: Noul(instructions=(
                f"chosen_action={chosen_label_for(c)!r}. proposed_command={command_for(elements[c])!r}. "
                f"target_bounds={elements[c].bounds!r}. {safe_instructions}"
            ))
            for key, c in safe_keys.items()
        },
    }
    answers = await jev.ask({"goal": goal, "candidates": criteria}, questions)
    pick = answers["pick"]
    fits = extract_fits(answers, labels)
    verdict = decide(pick.choice, pick.probabilities, pick.confidence, fits, labels)
    if not verdict.ok:
        return verdict, None
    command = CommandVariant(command=command_for(elements[verdict.choice]), rationale=f"{chosen_label_for(verdict.choice)} per the goal")
    safe_by_candidate = {c: answers[key].noul for key, c in safe_keys.items()}
    return verdict, finalize_gate(command, safe_by_candidate[verdict.choice])


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
    jev: JevClient, transport: AdbTransport, goal: str, *,
    parse_elements, pick_instructions: str, fit_instructions: str, safe_instructions: str,
    command_for, chosen_label_for, verbose: bool = True,
) -> TapProposal:
    """Shared by tap and long_press: narrow real matching elements + gate the resulting
    gesture. Only the element filter and command/label shape differ between callers."""
    dump_xml = await dump_screen(transport)
    elements = parse_elements(dump_xml)
    if verbose:
        print(f"{len(elements)} real matching elements on screen")

    cache_key = (foreground_package(dump_xml), goal)
    gate_result: GateResult | None = None

    async def _narrow() -> NarrowVerdict:
        nonlocal gate_result
        if elements and len(elements) <= CHUNK_SIZE:
            verdict, gate_result = await _fused_pick_and_gate(
                jev, goal, elements,
                pick_instructions=pick_instructions, fit_instructions=fit_instructions,
                safe_instructions=safe_instructions, command_for=command_for, chosen_label_for=chosen_label_for,
            )
            if verbose:
                print(f"Jev picked: {verdict.choice} (confidence {verdict.confidence:.2f}, fit {verdict.fit:.2f})")
        else:
            verdict = await narrow_and_pick(jev, goal, list(elements), instructions=pick_instructions, fit_instructions=fit_instructions, describe=lambda c: elements[c].description or c)
            if verbose:
                print(f"shortlist: {verdict.shortlist}")
                print(f"Jev picked: {verdict.choice} (confidence {verdict.confidence:.2f}, fit {verdict.fit:.2f})")
        return verdict

    verdict = await _cached_or_narrow(cache_key, elements, _narrow, verbose=verbose)

    if not verdict.ok:
        if verbose:
            print(f"=== ESCALATED === {'; '.join(verdict.reasons)}")
        return TapProposal(None, verdict.confidence, verdict.fit, reasons=tuple(verdict.reasons))

    target = elements[verdict.choice]
    chosen_label = chosen_label_for(verdict.choice)
    command = CommandVariant(command=command_for(target), rationale=f"{chosen_label} per the goal")
    if gate_result is None:
        gate_result = await gate_command(
            jev, command, chosen_label=chosen_label,
            evidence={"target_element": verdict.choice, "target_bounds": target.bounds},
            instructions=safe_instructions,
        )
    if verbose:
        print(f"gate verdict: {gate_result.verdict} ({gate_result.reason}, noul={gate_result.noul_confidence})")
    ready, pending, reasons = resolve_gate(gate_result, command, chosen_label)
    return TapProposal(verdict.choice, verdict.confidence, verdict.fit, ready, pending, reasons)


async def propose_tap(
    jev: JevClient, transport: AdbTransport, goal: str, *, verbose: bool = True,
    fit_instructions: str = "Would tapping {candidate} actually perform the goal?",
) -> TapProposal:
    """Narrow real on-screen elements + gate the resulting tap. No execution. `fit_instructions`
    defaults to a single-step framing ("actually perform the goal") -- a multi-step caller (see
    experiment/action_chain.py) can override it to ask about progress instead, since no single
    tap performs a whole multi-clause goal (confirmed live: fit 0.18 on a correctly-labeled
    real button, using the default wording, when the goal described several remaining steps)."""
    return await _propose_gesture(
        jev, transport, goal,
        parse_elements=parse_actionable_elements,
        pick_instructions="Which on-screen element would perform this goal?",
        fit_instructions=fit_instructions,
        safe_instructions=TAP_SAFE_INSTRUCTIONS,
        command_for=lambda el: f"input tap {el.x} {el.y}",
        chosen_label_for=lambda c: f"tap {c}",
        verbose=verbose,
    )


LONG_PRESS_SAFE_INSTRUCTIONS = (
    "Does the proposed_command's target match the chosen_action (same on-screen element, same "
    "location -- check target_bounds against the coordinates in proposed_command)? Answer no if "
    "it targets a different element or does anything beyond the chosen_action."
)


async def propose_long_press(
    jev: JevClient, transport: AdbTransport, goal: str, *, verbose: bool = True,
    fit_instructions: str = "Would long-pressing {candidate} actually perform the goal?",
) -> TapProposal:
    """Narrow real long-clickable elements + gate the resulting long-press. No execution --
    execute_tap runs it (a long-press is just `input swipe` with start==end). `fit_instructions`
    default is single-step framing; see propose_tap for why a multi-step caller overrides it."""
    return await _propose_gesture(
        jev, transport, goal,
        parse_elements=parse_long_clickable_elements,
        pick_instructions="Which on-screen element would this goal want long-pressed?",
        fit_instructions=fit_instructions,
        safe_instructions=LONG_PRESS_SAFE_INSTRUCTIONS,
        command_for=lambda el: f"input swipe {el.x} {el.y} {el.x} {el.y} 800",
        chosen_label_for=lambda c: f"long-press {c}",
        verbose=verbose,
    )


# The four real physical swipe directions a touchscreen has -- like DND_MODES/TOGGLEABLE_SERVICES,
# a genuinely fixed small set, not an app-specific menu of hardcoded options.
DIRECTIONS = {"up": None, "down": None, "left": None, "right": None}

SWIPE_SAFE_INSTRUCTIONS = (
    "Does the proposed_command's swipe direction match the chosen_action (same direction)? "
    "Answer no if it swipes a different direction or does anything beyond the chosen_action."
)


async def propose_swipe(jev: JevClient, transport: AdbTransport, goal: str, *, verbose: bool = True) -> ClosedSetProposal:
    """Pick one of the four real swipe directions for the goal + gate it. No execution."""
    width, height = await transport.window_size()
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
        pick_instructions=(
            "Which swipe direction does this goal want? 'scroll down'/'see more below' -> "
            "down; 'scroll up'/'go back up' -> up; likewise left/right for horizontal content."
        ),
        any_fit_instructions="Given direction_options, does any of them fit this goal?",
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


async def scroll_to_find(
    jev: JevClient, transport: AdbTransport, goal: str, *,
    direction: str = "down", max_attempts: int = 8, verbose: bool = True,
) -> ScrollToFindOutcome:
    """Re-checks the real screen against the goal every attempt and swipes only when the
    target genuinely isn't there yet -- reuses narrow_and_pick (is it visible now?) and
    propose_swipe/execute_swipe (one real gesture) instead of paging a fixed number of times."""
    for attempt in range(1, max_attempts + 1):
        elements = parse_all_elements(await dump_screen(transport))
        verdict = await narrow_and_pick(
            jev, goal, list(elements),
            instructions="Which real on-screen item is the target this goal is scrolling to find?",
            fit_instructions="Is {candidate} really the target this goal describes?",
            describe=lambda c: elements[c].description or c,
        )
        if verbose:
            print(f"attempt {attempt}/{max_attempts}: picked {verdict.choice!r} ok={verdict.ok}")
        if verdict.ok:
            return ScrollToFindOutcome(verdict.choice, attempt)

        proposal = await propose_swipe(jev, transport, f"scroll {direction}", verbose=False)
        # Only proposal.ready runs here -- a needs_approval verdict stops the loop like
        # any other unapproved command, it is never executed implicitly.
        if proposal.ready is None:
            return ScrollToFindOutcome(None, attempt, reasons=("swipe gate did not approve scrolling", *proposal.reasons))
        await execute_command(transport, proposal.ready)
    return ScrollToFindOutcome(None, max_attempts, reasons=(f"not found after {max_attempts} scrolls",))


async def _verify_after_action(
    jev: JevClient, transport: AdbTransport, goal: str, acted_on: str, *,
    delays: tuple[float, ...], verbose: bool,
) -> float:
    """Shared by execute_tap/execute_type: re-probe the real screen and ask
    Jev whether the goal is now achieved, retrying with backoff."""
    satisfied = 0.0
    for delay in delays:
        await asyncio.sleep(delay)
        # A fixed-length raw-XML truncation can cut off the real evidence entirely (confirmed
        # live); compact per-element labels carry far more real signal per character.
        screen_after = describe_screen(await dump_screen(transport), goal=goal)
        answers = await jev.ask(
            {"goal": goal, "acted_on": acted_on, "screen_after": screen_after},
            {"satisfied": Noul(instructions="screen_after lists every real element currently on "
                                             "screen, each described by its own real text/"
                                             "resource-id/content-desc. Given screen_after, is "
                                             "the goal now achieved?")},
        )
        satisfied = answers["satisfied"].noul
        if satisfied >= 0.5:
            break
    if verbose:
        print(f"goal met: {satisfied >= 0.5} (noul={satisfied:.2f})")
    return satisfied


async def _execute(
    jev: JevClient, transport: AdbTransport, goal: str, element: str, command: CommandVariant, *,
    confidence: float, verify: bool, delays: tuple[float, ...], verbose: bool,
) -> ActionOutcome:
    """Shared by execute_tap/execute_type: run the approved command, then verify
    unless the caller opts out (see verify=False docstring on either wrapper)."""
    await transport.run(command.command)
    if not verify:
        return ActionOutcome(element, confidence, True, 0.0, reasons=("verify skipped",))
    satisfied = await _verify_after_action(jev, transport, goal, element, delays=delays, verbose=verbose)
    return ActionOutcome(element, confidence, True, satisfied)


async def execute_tap(
    jev: JevClient, transport: AdbTransport, goal: str, element: str, command: CommandVariant, *,
    confidence: float = 0.0, verify: bool = True, delays: tuple[float, ...] = (0.0, 0.0, 0.0), verbose: bool = True,
) -> ActionOutcome:
    """Run an approved tap. verify=False skips the post-tap dump+Jev-ask
    (~2.9s) for a caller that will check the resulting state itself -- e.g.
    an agent sequencing several steps before its own screenshot/inspection."""
    return await _execute(jev, transport, goal, element, command, confidence=confidence, verify=verify, delays=delays, verbose=verbose)


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
    return await jev.ask(
        {"goal": goal, "candidate_values": spans},
        {
            "value": Choice(instructions="Which of these is the text to type for this goal?", criteria={s: None for s in spans}),
            "any_fit": Noul(instructions="Given candidate_values, does any of them belong in a text field for this goal?"),
        },
    )


async def _fused_field_and_value(
    jev: JevClient, goal: str, elements: dict[str, Element], spans: list[str], fit_instructions: str,
) -> tuple[NarrowVerdict, dict | None]:
    """Batches field-narrow with value-pick into one real request on a cache miss -- they're
    independent facts that previously cost two separate round trips (asyncio.gather only
    overlaps wall-clock time, it doesn't merge the payloads)."""
    labels = list(elements)
    field_criteria = {c: None for c in labels}
    state = {"goal": goal, "candidates": field_criteria}
    questions = {
        "pick": Choice(instructions="Which on-screen field should receive text for this goal?", criteria=field_criteria),
        **fit_questions(labels, fit_instructions, lambda c: elements[c].description or c),
    }
    if spans:
        state["candidate_values"] = spans
        questions["value"] = Choice(instructions="Which of these is the text to type for this goal?", criteria={s: None for s in spans})
        questions["any_fit_value"] = Noul(instructions="Given candidate_values, does any of them belong in a text field for this goal?")
    answers = await jev.ask(state, questions)
    pick = answers["pick"]
    fits = extract_fits(answers, labels)
    # A field's label embeds its live text, so it reads as a new candidate once typed into --
    # same fix as run_dumpsys_query's answer-field pick: min_fit is the real bar on a small pool.
    loose = {"min_confidence": 0.0, "min_margin": 0.0} if len(elements) <= 3 else {}
    field_verdict = decide(pick.choice, pick.probabilities, pick.confidence, fits, labels, **loose)
    value_answers = {"value": answers["value"], "any_fit": answers["any_fit_value"]} if spans else None
    return field_verdict, value_answers


async def propose_type(
    jev: JevClient, transport: AdbTransport, goal: str, *, verbose: bool = True,
    fit_instructions: str = "Would typing into {candidate} actually serve the goal?",
) -> TypeProposal:
    """Narrow real editable fields + extract/pick the real value to type + gate. No execution.
    `fit_instructions` default is single-step framing; see propose_tap for why a multi-step
    caller overrides it."""
    dump_xml = await dump_screen(transport)
    elements = parse_editable_elements(dump_xml)
    if verbose:
        print(f"{len(elements)} real editable elements on screen")
    if not elements:
        return TypeProposal(None, None, 0.0, reasons=_no_editable_field_reasons(dump_xml, goal))

    spans = extract_value_spans(goal)
    cache_key = (foreground_package(dump_xml), goal)
    cached = _ELEMENT_CACHE.get(cache_key)
    if cached in elements:
        if verbose:
            print(f"cache hit: {cached!r} still on screen, skipping Jev narrowing")
        field_verdict = NarrowVerdict(cached, 1.0, 1.0, 1.0, [cached])
        answers = await _pick_value(jev, goal, spans)
    else:
        field_verdict, answers = await _fused_field_and_value(jev, goal, elements, spans, fit_instructions)
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
    if answers["any_fit"].noul < 0.5:
        return TypeProposal(field_verdict.choice, None, field_verdict.confidence, reasons=("no candidate value fits a text field for this goal",))
    value = answers["value"].choice

    target = elements[field_verdict.choice]
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
        print(f"gate verdict: {gate_result.verdict} ({gate_result.reason}, noul={gate_result.noul_confidence})")
    ready, pending, reasons = resolve_gate(gate_result, command, chosen_label)
    return TypeProposal(field_verdict.choice, value, field_verdict.confidence, ready, pending, reasons)


async def execute_type(
    jev: JevClient, transport: AdbTransport, goal: str, element: str, command: CommandVariant, *,
    confidence: float = 0.0, verify: bool = True, delays: tuple[float, ...] = (0.0, 0.0, 0.0), verbose: bool = True,
) -> ActionOutcome:
    """Run an approved type-into-field action. verify=False skips the
    post-action dump+Jev-ask, same tradeoff as execute_tap."""
    return await _execute(jev, transport, goal, element, command, confidence=confidence, verify=verify, delays=delays, verbose=verbose)
