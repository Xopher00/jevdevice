"""Propose/execute for real Android system-service actions: toggling a radio, pressing a key,
setting Do Not Disturb, and reading a dumpsys service's status. Each fills a closed-set argument
(Jev's Choice) or a real enumeration (dumpsys -l), then code builds and gates the command --
the same pattern ui.py uses for on-screen elements, applied to system services instead.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from .common import gated
from .gate import CommandVariant, Pending, gate_command, resolve_gate
from .jev import Choice, JevClient, Noul
from .narrowing import narrow_and_pick
from .transport import AdbTransport

# svc's controllable services are a small, genuinely fixed set (not app-specific).
TOGGLEABLE_SERVICES = {"bluetooth": None, "nfc": None, "data": None}

# Real KEYCODE_* names -- `input keyevent` accepts these directly (confirmed live),
# no hand-maintained number table needed. POWER excluded: can lock/reboot the device.
KEY_EVENTS = (
    "HOME", "BACK", "APP_SWITCH", "ENTER",
    "VOLUME_UP", "VOLUME_DOWN", "VOLUME_MUTE",
    "MEDIA_PLAY_PAUSE", "MEDIA_NEXT", "MEDIA_PREVIOUS",
    "CAMERA",
)

# cmd notification's real, OS-fixed Do Not Disturb modes.
DND_MODES = {"off": None, "priority": None, "alarms": None, "none": None}

# Not yet calibrated live (see calibrate/gate_taps.py for the batched-comparison
# method used for TAP_SAFE_INSTRUCTIONS) -- domain-adapted from the same pattern in the meantime.
KEYEVENT_SAFE_INSTRUCTIONS = (
    "Does the proposed_command's key press match the chosen_action (same key)? "
    "Answer no if it presses a different key or does anything beyond the chosen_action."
)
DND_SAFE_INSTRUCTIONS = (
    "Does the proposed_command's Do Not Disturb mode match the chosen_action (same mode)? "
    "Answer no if it sets a different mode or does anything beyond the chosen_action."
)


def parse_dumpsys_services(raw: str) -> list[str]:
    """Excludes HAL/AIDL binder interfaces (registered as
    `reverse.dotted.pkg.IInterface/instance`, confirmed against a real
    `dumpsys -l` dump) so they never compete with their own plain
    system-manager service (e.g. `battery`) for the same real subsystem --
    structural, from the name's shape, not a curated list of which ones."""
    return [line.strip() for line in raw.splitlines()
            if line.strip() and not line.strip().endswith(":") and "/" not in line]


_NAME_VALUE = re.compile(r"(?:^|\s)name:(\S+)"), re.compile(r"(?:^|\s)value:(\S+)")


def parse_key_value(raw: str) -> dict[str, str]:
    """Most dumpsys services print one `key: value` per line (battery, wifi). `dumpsys
    settings` instead prints several real `name:X ... value:Y` pairs on one line per entry
    (e.g. `_id:2006 name:zen_mode pkg:android value:2 ...`) -- extract that shape directly
    rather than mis-keying on the line's first token, which loses the real name entirely."""
    name_re, value_re = _NAME_VALUE
    result: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        name_match, value_match = name_re.search(line), value_re.search(line)
        if name_match and value_match:
            result[name_match.group(1)] = value_match.group(1)
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key and value and " " not in key:
            result[key] = value
    return result


@dataclass
class ToggleProposal:
    """Result of filling arguments + gating a toggle goal, before any mutation runs."""
    service: str | None
    enabled: str | None
    ready: CommandVariant | None = None
    pending: Pending | None = None
    reasons: tuple[str, ...] = ()


async def propose_toggle(jev: JevClient, transport: AdbTransport, goal: str, *, verbose: bool = True) -> ToggleProposal:
    if verbose:
        print("--- step 2: Jev fills the two closed-set arguments (batched) ---")
    answers = await jev.ask(
        {"goal": goal, "radio_options": TOGGLEABLE_SERVICES},
        {
            "service": Choice(instructions="Which service does the goal refer to?", criteria=TOGGLEABLE_SERVICES),
            "enabled": Choice(instructions="Does the goal want it turned on or off?", criteria={"on": None, "off": None}),
            "names_one": Noul(instructions="Given radio_options, does the goal specifically ask about one of them?"),
        },
    )
    service_pick, enabled_pick = answers["service"], answers["enabled"]
    if verbose:
        print(f"service: {service_pick.choice} (confidence {service_pick.confidence:.2f})")
        print(f"enabled: {enabled_pick.choice} (confidence {enabled_pick.confidence:.2f})")
        print(f"names_one: {answers['names_one'].noul:.2f}\n")
    if answers["names_one"].noul < 0.5:
        return ToggleProposal(None, None, reasons=(f"goal doesn't name one of {', '.join(TOGGLEABLE_SERVICES)}",))
    service, enabled = gated(service_pick), gated(enabled_pick)
    if service is None or enabled is None:
        return ToggleProposal(service, enabled, reasons=("confidence gate rejected service or enabled pick",))

    verb = "enable" if enabled == "on" else "disable"
    command = CommandVariant(command=f"svc {service} {verb}", rationale=f"{verb} {service} per the goal")
    chosen_label = f"{verb} {service}"
    if verbose:
        print(f"--- step 3: gate (deny-list + Jev Noul) before executing: {command.command!r} ---")
    gate_result = await gate_command(jev, command, chosen_label=chosen_label)
    if verbose:
        print(f"gate verdict: {gate_result.verdict} ({gate_result.reason}, noul={gate_result.noul_confidence})\n")
    ready, pending, reasons = resolve_gate(gate_result, command, chosen_label)
    return ToggleProposal(service, enabled, ready, pending, reasons)


@dataclass
class ToggleOutcome:
    exit_code: int
    check_service: str | None
    resolve_confidence: float
    satisfied: float
    reasons: tuple[str, ...] = ()


async def execute_toggle(
    jev: JevClient, transport: AdbTransport, goal: str, service: str, command: CommandVariant, *, verbose: bool = True,
) -> ToggleOutcome:
    if verbose:
        print("--- step 4/5: real mutating command + real service enumeration (independent, run concurrently) ---")
    result, services_raw = await asyncio.gather(transport.run(command.command), transport.run("dumpsys -l"))
    if verbose:
        print(f"exit_code={result.exit_code}")
    services = parse_dumpsys_services(services_raw.stdout)
    resolve_verdict = await narrow_and_pick(
        jev, goal, services,
        instructions="Which dumpsys service would show this toggled service's real status?",
        fit_instructions="Would `dumpsys {candidate}` actually show " + service + "'s real status?",
        state_extra={"toggled_service": service},
    )
    if verbose:
        print(f"shortlist: {resolve_verdict.shortlist}")
    if not resolve_verdict.ok:
        if verbose:
            print(f"could not resolve a status service to verify with -- unverified ({'; '.join(resolve_verdict.reasons)})")
        return ToggleOutcome(result.exit_code, None, resolve_verdict.confidence, 0.0, tuple(resolve_verdict.reasons))
    check_service = resolve_verdict.choice
    if verbose:
        print(f"checking real service: {check_service!r} (confidence {resolve_verdict.confidence:.2f}, fit {resolve_verdict.fit:.2f})\n")

    if verbose:
        print("--- step 6: real check + Jev verify ---")
    check = await transport.run(f"dumpsys {check_service}")
    verify = await jev.ask(
        {"goal": goal, "service_state": check.stdout[:2000]},
        {"satisfied": Noul(instructions="Given service_state, is the goal now achieved?")},
    )
    satisfied = verify["satisfied"].noul
    if verbose:
        print(f"\n=== RESULT ===\nJev verify noul: {satisfied:.2f}\ngoal met: {satisfied >= 0.5}")
    return ToggleOutcome(result.exit_code, check_service, resolve_verdict.confidence, satisfied)


@dataclass
class KeyEventProposal:
    key: str | None
    confidence: float
    ready: CommandVariant | None = None
    pending: Pending | None = None
    reasons: tuple[str, ...] = ()


async def propose_keyevent(jev: JevClient, goal: str, *, verbose: bool = True) -> KeyEventProposal:
    answers = await jev.ask(
        {"goal": goal, "key_options": dict.fromkeys(KEY_EVENTS)},
        {
            "key": Choice(instructions="Which key/button press would achieve this goal?", criteria=dict.fromkeys(KEY_EVENTS)),
            "any_fit": Noul(instructions="Given key_options, does any of them fit this goal?"),
        },
    )
    key_pick = answers["key"]
    if verbose:
        print(f"picked key: {key_pick.choice} (confidence {key_pick.confidence:.2f}, any_fit {answers['any_fit'].noul:.2f})")
    if answers["any_fit"].noul < 0.5:
        return KeyEventProposal(None, key_pick.confidence, reasons=(f"none of the {len(KEY_EVENTS)} keys fit this goal",))
    key = gated(key_pick)
    if key is None:
        return KeyEventProposal(None, key_pick.confidence, reasons=("confidence gate rejected the key pick",))

    command = CommandVariant(command=f"input keyevent KEYCODE_{key}", rationale=f"press {key} per the goal")
    chosen_label = f"press {key}"
    gate_result = await gate_command(jev, command, chosen_label=chosen_label, instructions=KEYEVENT_SAFE_INSTRUCTIONS)
    if verbose:
        print(f"gate verdict: {gate_result.verdict} ({gate_result.reason}, noul={gate_result.noul_confidence})")
    ready, pending, reasons = resolve_gate(gate_result, command, chosen_label)
    return KeyEventProposal(key, key_pick.confidence, ready, pending, reasons)


async def execute_keyevent(transport: AdbTransport, command: CommandVariant) -> int:
    result = await transport.run(command.command)
    return result.exit_code


async def take_screenshot(transport: AdbTransport) -> bytes:
    """No candidates, no ambiguity, always safe (screencap is in
    gate.READ_ONLY_PREFIXES) -- same reasoning dumpsys uses to skip the gate
    entirely. exec-out streams the PNG on stdout; nothing is written to the device."""
    return await transport.run_binary("screencap -p")


@dataclass
class DndProposal:
    mode: str | None
    confidence: float
    ready: CommandVariant | None = None
    pending: Pending | None = None
    reasons: tuple[str, ...] = ()


async def propose_dnd(jev: JevClient, goal: str, *, verbose: bool = True) -> DndProposal:
    answers = await jev.ask(
        {"goal": goal, "mode_options": DND_MODES},
        {
            "mode": Choice(instructions="Which Do Not Disturb mode does this goal want?", criteria=DND_MODES),
            "any_fit": Noul(instructions="Given mode_options, does any of them fit this goal?"),
        },
    )
    mode_pick = answers["mode"]
    if verbose:
        print(f"picked DND mode: {mode_pick.choice} (confidence {mode_pick.confidence:.2f}, any_fit {answers['any_fit'].noul:.2f})")
    if answers["any_fit"].noul < 0.5:
        return DndProposal(None, mode_pick.confidence, reasons=(f"none of the {len(DND_MODES)} DND modes fit this goal",))
    mode = gated(mode_pick)
    if mode is None:
        return DndProposal(None, mode_pick.confidence, reasons=("confidence gate rejected the DND mode pick",))

    command = CommandVariant(command=f"cmd notification set_dnd {mode}", rationale=f"set Do Not Disturb to {mode} per the goal")
    chosen_label = f"set Do Not Disturb to {mode}"
    gate_result = await gate_command(jev, command, chosen_label=chosen_label, instructions=DND_SAFE_INSTRUCTIONS)
    if verbose:
        print(f"gate verdict: {gate_result.verdict} ({gate_result.reason}, noul={gate_result.noul_confidence})")
    ready, pending, reasons = resolve_gate(gate_result, command, chosen_label)
    return DndProposal(mode, mode_pick.confidence, ready, pending, reasons)


async def execute_dnd(transport: AdbTransport, command: CommandVariant) -> int:
    result = await transport.run(command.command)
    return result.exit_code


@dataclass
class DumpsysOutcome:
    service: str | None
    confidence: float
    fit: float
    parsed: dict[str, str] | None = None
    answer_key: str | None = None
    reasons: tuple[str, ...] = ()


async def run_dumpsys_query(jev: JevClient, transport: AdbTransport, goal: str, *, verbose: bool = True) -> DumpsysOutcome:
    if verbose:
        print("--- step 2: real probe for the open argument's real enumeration ---")
    services_raw = await transport.run("dumpsys -l")
    services = parse_dumpsys_services(services_raw.stdout)
    if verbose:
        print(f"parsed {len(services)} real dumpsys services (no model)\n")
        print("--- step 3: two-round semantic narrow over the real enumeration ---")

    verdict = await narrow_and_pick(
        jev, goal, services,
        instructions="Which service would answer this goal?",
        fit_instructions="Would `dumpsys {candidate}` actually contain the answer to the goal?",
        accept_any_fitting=True,
    )
    if verbose:
        print(f"shortlist: {verdict.shortlist}")
        print(f"picked service: {verdict.choice} (confidence {verdict.confidence:.2f}, fit {verdict.fit:.2f})\n")
    if not verdict.ok:
        if verbose:
            print(f"=== ESCALATED === {'; '.join(verdict.reasons)}")
        return DumpsysOutcome(None, verdict.confidence, verdict.fit, reasons=tuple(verdict.reasons))

    if verbose:
        print(f"--- step 4: code builds the command deterministically: dumpsys {verdict.choice} ---")
    result = await transport.run(f"dumpsys {verdict.choice}")
    parsed = parse_key_value(result.stdout)
    if verbose:
        print(f"parsed {len(parsed)} key:value pairs (no model)\n")

    answer_key = None
    if parsed:
        async def _field_values(shortlist: list[str]) -> dict[str, str]:
            return {c: parsed[c] for c in shortlist}

        # min_fit is the real bar for this read-only pick -- raw Choice confidence spreads
        # thin over 200+ plausible field names even for a clearly-correct answer. evidence_for
        # attaches real values only to the round-2 shortlist, not every round-1 chunk --
        # dumpsys settings alone can parse to 2000+ real fields.
        field_verdict = await narrow_and_pick(
            jev, goal, list(parsed),
            instructions="Which real field would answer the goal?",
            fit_instructions="Does the field {candidate} actually answer the goal?",
            evidence_for=_field_values,
            accept_any_fitting=True,
        )
        if field_verdict.ok:
            answer_key = field_verdict.choice
    if verbose:
        print("=== RESULT ===")
        for key, value in list(parsed.items())[:8]:
            print(f"  {key}: {value}")
        if answer_key:
            print(f"\nanswer: {answer_key} = {parsed[answer_key]}")
    return DumpsysOutcome(verdict.choice, verdict.confidence, verdict.fit, parsed, answer_key)
