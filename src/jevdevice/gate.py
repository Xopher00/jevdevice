"""Gate: every mutating command passes a deny-list, then a Jev Noul, before it
runs. Classification never trusts what the small model calls the command —
is_read_only/is_denied are applied to the literal command string regardless
of any label attached to it.
"""

from __future__ import annotations

import re
import shlex
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

from .budget import choice_criteria, current_profile, is_abstain
from .jev import Choice, JevClient, Noul
from .matching import confidence_gate

# Real argv shapes classified as read-only, by (argv[0], rest-of-argv-prefix-or-None).
# None means "any args" -- e.g. dumpsys is read-only regardless of which service.
_READ_ONLY_ARGV = (
    ("dumpsys", None), ("getprop", None), ("screencap", None),
    ("settings", ("get",)), ("cmd", ("package", "resolve-activity")),
    ("uiautomator", ("dump",)), ("wm", ("size",)), ("wm", ("density",)),
    ("input", ("keyevent", "KEYCODE_HOME")),  # pure navigation, never changes app/device state
)
_READ_ONLY_PM_SUBCOMMANDS = ("list", "dump", "path", "resolve")
# cat/ls/echo need a real argument to be meaningfully read-only, matching the
# historical "prefix + trailing space" shape they were classified by.
_READ_ONLY_HEADS_NEEDING_ARGS = ("cat", "ls", "echo")

# Real argv shapes classified as denied, by (argv[0], rest-of-argv-prefix-or-None), case-folded.
_DENY_ARGV = (
    ("pm", ("clear",)), ("pm", ("uninstall",)), ("rm", None), ("reboot", None),
    ("su", None), ("dd", None), ("format", None), ("chmod", ("777",)),
    ("settings", ("put", "secure")), ("settings", ("put", "global")),
    # wifi disable over wireless adb severs the connection this tool needs to run anything else.
    ("svc", ("wifi", "disable")), ("svc", ("wifi", "0")),
)
_DENY_WORDS = ("wipe", "factory")  # matched as a whole word, so "swipe"/"unfactored" never collide
_WORD_RE = re.compile(r"[a-z0-9]+")


def _invocations(command: str) -> list[list[str]] | None:
    """Real argv per `&&`-chained sub-command, from the actual shell shape the
    codebase builds (see ui.py's tap+type chaining) -- not a text split, so a
    quoted value that happens to contain the literal text "&&" stays one token."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    invocations: list[list[str]] = [[]]
    for token in tokens:
        if token == "&&":
            invocations.append([])
        else:
            invocations[-1].append(token)
    return invocations if all(invocations) else None


def _matches_argv(argv: list[str], head: str, rest: tuple[str, ...] | None) -> bool:
    if argv[0] != head:
        return False
    return rest is None or tuple(argv[1:1 + len(rest)]) == rest


def _is_read_only_argv(argv: list[str]) -> bool:
    if argv[0] == "pm":
        return len(argv) >= 2 and argv[1] in _READ_ONLY_PM_SUBCOMMANDS
    if argv[0] in _READ_ONLY_HEADS_NEEDING_ARGS:
        return len(argv) >= 2
    return any(_matches_argv(argv, head, rest) for head, rest in _READ_ONLY_ARGV)


def _is_denied_argv(argv: list[str]) -> bool:
    folded = [token.casefold() for token in argv]
    if any(_matches_argv(folded, head, rest) for head, rest in _DENY_ARGV):
        return True
    if any(word in _WORD_RE.findall(token) for token in folded for word in _DENY_WORDS):
        return True
    # `> /dev/...` redirection: shlex keeps ">" as its own token when it's bare in the string.
    return any(tok == ">" and nxt.startswith("/dev/") for tok, nxt in pairwise(folded))


def is_read_only(command: str) -> bool:
    invocations = _invocations(command)
    if not invocations:
        return False
    return all(_is_read_only_argv(argv) for argv in invocations)


def is_denied(command: str) -> bool:
    invocations = _invocations(command)
    if not invocations:
        return False
    return any(_is_denied_argv(argv) for argv in invocations)


# calibrate/gate.py-verified: 0.82-0.94 for correct commands, 0.05 for a wrong target.
DEFAULT_SAFE_INSTRUCTIONS = (
    "Does the proposed_command's target and effect match the chosen_action "
    "(same service/component, same on-or-off direction)? Answer no if it names a "
    "different target, a different effect, or chains on any additional command."
)


@dataclass
class CommandVariant:
    command: str
    rationale: str


class GateVerdict:
    APPROVED = "approved"
    DENIED = "denied"
    NEEDS_APPROVAL = "needs_approval"


@dataclass
class GateResult:
    verdict: str
    reason: str
    noul_confidence: float | None = None
    # Journal linkage: the ask() call whose answer produced this verdict, so the
    # approval/execution flow (mcp_server PendingAction) can join its outcome row
    # to the gate's decision row. None for verdicts settled without an ask
    # (read_only/deny_listed fast paths).
    call_id: str | None = None


@dataclass
class Pending:
    """A gated command awaiting a decision -- carries everything needed to finish
    later, so approval can be a separate step instead of a blocking prompt inline
    in the core pipeline. CLI code resolves this with confirm_with_human(); an MCP
    server resolves it with a second tool call instead. Core logic doesn't choose."""
    command: CommandVariant
    chosen_label: str
    gate_result: GateResult


def resolve_gate(gate_result: GateResult, command: CommandVariant, chosen_label: str) -> tuple[CommandVariant | None, Pending | None, tuple[str, ...]]:
    """The APPROVED/NEEDS_APPROVAL/DENIED -> (ready, pending, reasons) mapping
    every propose_* function needs -- one place instead of five copies."""
    if gate_result.verdict == GateVerdict.APPROVED:
        return command, None, ()
    if gate_result.verdict == GateVerdict.NEEDS_APPROVAL:
        return None, Pending(command, chosen_label, gate_result), ()
    return None, None, (gate_result.reason,)


def confirm_with_human(command: CommandVariant, chosen_label: str, noul_confidence: float | None) -> bool:
    """needs_approval means defer to a person, not silently drop the action."""
    conf = f"{noul_confidence:.2f}" if noul_confidence is not None else "n/a"
    print(f"\nApprove this command? (Jev confidence {conf})\n  action: {chosen_label}\n  command: {command.command}")
    try:
        return input("  [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


async def gate_command(
    jev: JevClient, command: CommandVariant, chosen_label: str, threshold: float | None = None,
    evidence: dict | None = None, instructions: str = DEFAULT_SAFE_INSTRUCTIONS,
    *, call_id: str | None = None,
) -> GateResult:
    """`evidence` is real, code-verified data backing the command (e.g. the
    tapped element's actual on-screen bounds) -- something the model can check
    itself, not another description it has to take on faith like `rationale`.
    `instructions` defaults to the toggle-command wording; a different command
    shape (e.g. a tap) needs its own calibrated wording, not this one reused.
    `threshold` defaults to the answering engine's profile knob (budget.py):
    each engine's noul scale is its own, so the approval floor travels with it."""
    if is_read_only(command.command):
        return GateResult(verdict=GateVerdict.APPROVED, reason="read_only")
    if is_denied(command.command):
        return GateResult(verdict=GateVerdict.DENIED, reason="deny_listed")
    profile = current_profile(jev.engine_name)
    threshold = profile.gate_threshold if threshold is None else threshold

    # The call_id is generated HERE (not inside ask()) so it can travel onto the
    # GateResult -> Pending -> PendingAction -> outcome row and join the two rows.
    call_id = call_id or str(uuid.uuid4())
    answers = await jev.ask(
        {"chosen_action": chosen_label, "proposed_command": command.command, "rationale": command.rationale,
         **(evidence or {})},
        {"safe": Noul(instructions=instructions)},
        phase="gate", call_id=call_id,
    )
    return finalize_gate(command, answers["safe"].noul, threshold, call_id=call_id)


def finalize_gate(
    command: CommandVariant, confidence: float, threshold: float = 0.8, *, call_id: str | None = None,
) -> GateResult:
    """Applies the same deny-list + threshold rule gate_command does, but takes an
    already-computed confidence -- for callers that got it speculatively (e.g. asked
    alongside a narrowing pick in one batched request) instead of via their own ask.
    Callers with a client pass the answering engine's profile.gate_threshold
    (gate_command resolves it that way); the bare default stays the jev-era 0.8.
    `call_id` joins the verdict to that shared ask's decision row (see _fused_pick_and_gate)."""
    if is_denied(command.command):
        return GateResult(verdict=GateVerdict.DENIED, reason="deny_listed")
    if confidence >= threshold:
        return GateResult(verdict=GateVerdict.APPROVED, reason="jev_confirmed", noul_confidence=confidence, call_id=call_id)
    return GateResult(verdict=GateVerdict.NEEDS_APPROVAL, reason="jev_uncertain", noul_confidence=confidence, call_id=call_id)


@dataclass
class ClosedSetProposal:
    """Result of a Jev Choice over one fixed small option set (a key, a DND mode, a swipe
    direction) plus that pick's gate, before any mutation runs. toggle_service makes two
    picks (service + on/off) and keeps its own proposal type."""
    pick: str | None
    confidence: float
    ready: CommandVariant | None = None
    pending: Pending | None = None
    reasons: tuple[str, ...] = ()
    gate_result: GateResult | None = None  # journal linkage: outcome rows read gate_result.call_id


async def propose_from_closed_set(
    jev: JevClient, goal: str, options: dict[str, str | None], *,
    options_key: str, pick_instructions: str, any_fit_instructions: str,
    command_for: Callable[[str], CommandVariant], label_for: Callable[[str], str],
    gate_instructions: str, verbose: bool = True, pick_verb: str = "pick",
) -> ClosedSetProposal:
    """Shared propose pipeline for closed-set actions (keyevent, DND, swipe): one batched
    ask -- which option fits, and does any of them fit -- then the deterministic command
    built from the winning option goes through gate_command. Never executes; returns
    reasons when nothing fits, the judge abstains (none_of_these), or the pick fails
    the confidence gate."""
    profile = current_profile(jev.engine_name)
    answers = await jev.ask(
        {"goal": goal, options_key: options},
        {
            "pick": Choice(instructions=pick_instructions, criteria=choice_criteria(options, profile)),
            "any_fit": Noul(instructions=any_fit_instructions),
        },
        phase="fill",
    )
    pick_answer = answers["pick"]
    if verbose:
        print(f"picked {pick_verb}: {pick_answer.choice} (confidence {pick_answer.confidence:.2f}, any_fit {answers['any_fit'].noul:.2f})")
    if is_abstain(pick_answer.choice):
        return ClosedSetProposal(None, pick_answer.confidence, reasons=("judge abstained: picked none_of_these, so none of the options fit",))
    if answers["any_fit"].noul < profile.noul_floor:
        return ClosedSetProposal(None, pick_answer.confidence, reasons=(f"none of the {len(options)} options fit this goal",))
    ok, reason = confidence_gate(pick_answer.probabilities, pick_answer.confidence, profile.min_confidence, profile.min_margin)
    if not ok:
        return ClosedSetProposal(None, pick_answer.confidence, reasons=(reason,))
    command = command_for(pick_answer.choice)
    chosen_label = label_for(pick_answer.choice)
    gate_result = await gate_command(jev, command, chosen_label=chosen_label, instructions=gate_instructions)
    if verbose:
        print(f"gate verdict: {gate_result.verdict} ({gate_result.reason}, noul={gate_result.noul_confidence})")
    ready, pending, reasons = resolve_gate(gate_result, command, chosen_label)
    return ClosedSetProposal(pick_answer.choice, pick_answer.confidence, ready, pending, reasons, gate_result=gate_result)
