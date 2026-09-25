"""Propose/execute for real Android system-service actions: toggling a radio, pressing a key,
setting Do Not Disturb, and reading a dumpsys service's status. Each fills a closed-set argument
(Jev's Choice) or a real enumeration (dumpsys -l), then code builds and gates the command --
the same pattern ui.py uses for on-screen elements, applied to system services instead.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from jevdevice import question_sets
from jevdevice.budget import choice_criteria, current_profile, is_abstain
from jevdevice.common import gated
from jevdevice.device import Device
from jevdevice.jev import JudgeEngine, ask
from jevdevice.judge.gate import (
    ClosedSetProposal,
    CommandVariant,
    GateResult,
    Pending,
    gate_command,
    propose_from_closed_set,
    resolve_gate,
)
from jevdevice.judge.narrowing import narrow_and_pick

# svc's controllable services are a small, genuinely fixed set (not app-specific).
TOGGLEABLE_SERVICES = {"bluetooth": None, "nfc": None, "data": None}

# Real KEYCODE_* names -- `input keyevent` accepts these directly,
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
# The artifact is the single source of truth.
KEYEVENT_SAFE_INSTRUCTIONS = question_sets.text("keyevent.safe")
DND_SAFE_INSTRUCTIONS = question_sets.text("dnd.safe")


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
    gate_result: GateResult | None = None  # journal linkage: outcome rows read gate_result.call_id
    pick_call_id: str | None = None  # the "service"/"enabled" answers' own row -- device verdicts label this
    label_keys: tuple[str, ...] = ()  # a closed-set service pick has no fit
    gate_key: str | None = None  # "safe" when the gate ask ran


async def propose_toggle(jev: JudgeEngine, device: Device, goal: str, *, verbose: bool = True) -> ToggleProposal:
    if verbose:
        print("--- step 2: Jev fills the two closed-set arguments (batched) ---")
    profile = current_profile(jev.name)
    call_id, answers = await ask(
        jev,
        {"goal": goal, "radio_options": TOGGLEABLE_SERVICES},
        {
            "service": question_sets.choice("toggle.service", choice_criteria(TOGGLEABLE_SERVICES, profile)),
            "enabled": question_sets.choice("toggle.enabled", choice_criteria({"on": None, "off": None}, profile)),
            "names_one": question_sets.noul("toggle.names_one"),
        },
        phase="fill",
    )
    service_pick, enabled_pick = answers["service"], answers["enabled"]
    if verbose:
        print(f"service: {service_pick.choice} (confidence {service_pick.confidence:.2f})")
        print(f"enabled: {enabled_pick.choice} (confidence {enabled_pick.confidence:.2f})")
        print(f"names_one: {answers['names_one'].noul:.2f}\n")
    if answers["names_one"].noul < profile.noul_floor:
        return ToggleProposal(None, None, reasons=(f"goal doesn't name one of {', '.join(TOGGLEABLE_SERVICES)}",))
    if is_abstain(service_pick.choice) or is_abstain(enabled_pick.choice):
        return ToggleProposal(None, None, reasons=("judge abstained: picked none_of_these for the service or its direction",))
    service, enabled = gated(service_pick, profile=profile), gated(enabled_pick, profile=profile)
    if service is None or enabled is None:
        return ToggleProposal(service, enabled, reasons=("confidence gate rejected service or enabled pick",))

    verb = "enable" if enabled == "on" else "disable"
    command = CommandVariant(command=f"svc {service} {verb}", rationale=f"{verb} {service} per the goal")
    chosen_label = f"{verb} {service}"
    if verbose:
        print(f"--- step 3: gate (deny-list + Jev Noul) before executing: {command.command!r} ---")
    gate_result = await gate_command(jev, command, chosen_label=chosen_label)
    if verbose:
        print(f"gate verdict: {gate_result.verdict} ({gate_result.reason}, noul={gate_result.confidence})\n")
    ready, pending, reasons = resolve_gate(gate_result, command, chosen_label)
    return ToggleProposal(service, enabled, ready, pending, reasons,
                          gate_result=gate_result, pick_call_id=call_id,
                          gate_key="safe" if gate_result.call_id is not None else None)


@dataclass
class ToggleOutcome:
    exit_code: int
    check_service: str | None
    resolve_confidence: float
    satisfied: float
    reasons: tuple[str, ...] = ()


async def execute_toggle(
    jev: JudgeEngine, device: Device, goal: str, service: str, command: CommandVariant,
    *, verbose: bool = True,
) -> ToggleOutcome:
    if verbose:
        print("--- step 4/5: real mutating command + real service enumeration (independent, run concurrently) ---")
    result, services_raw = await asyncio.gather(device.run(command.command), device.run("dumpsys -l"))
    if verbose:
        print(f"exit_code={result.exit_code}")
    services = parse_dumpsys_services(services_raw.stdout)
    resolve_verdict = await narrow_and_pick(
        jev, goal, services,
        instructions=question_sets.text("toggle.resolve_status.pick"),
        fit_instructions=question_sets.text("toggle.resolve_status.fit", toggled_service=service),
        state_extra={"toggled_service": service}, pick_qid="toggle.resolve_status.pick",
        fit_qid="toggle.resolve_status.fit",
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
    check = await device.run(f"dumpsys {check_service}")
    # The raw status text rides in state under the answering engine's profile
    # budget -- the clip is a named knob (budget.py), never a bare int.
    _, verify = await ask(
        jev,
        {"goal": goal, "service_state": check.stdout[:current_profile(jev.name).probe_max_chars]},
        {"satisfied": question_sets.noul("toggle.verify_satisfied")},
        phase="verify",
    )
    satisfied = verify["satisfied"].noul
    if verbose:
        print(f"\n=== RESULT ===\nJev verify noul: {satisfied:.2f}\ngoal met: {satisfied >= current_profile(jev.name).noul_floor}")
    return ToggleOutcome(result.exit_code, check_service, resolve_verdict.confidence, satisfied)


async def propose_keyevent(jev: JudgeEngine, goal: str, *, verbose: bool = True) -> ClosedSetProposal:
    return await propose_from_closed_set(
        jev, goal, dict.fromkeys(KEY_EVENTS),
        options_key="key_options",
        pick_instructions=question_sets.text("keyevent.pick"),
        any_fit_instructions=question_sets.text("keyevent.any_fit"),
        pick_verb="key",
        command_for=lambda key: CommandVariant(command=f"input keyevent KEYCODE_{key}", rationale=f"press {key} per the goal"),
        label_for=lambda key: f"press {key}",
        gate_instructions=KEYEVENT_SAFE_INSTRUCTIONS,
        verbose=verbose, pick_qid="keyevent.pick", any_fit_qid="keyevent.any_fit", gate_qid="keyevent.safe",
    )


async def execute_command(device: Device, command: CommandVariant) -> int:
    """Run an already-gated single command; the exit code is all there is to report.
    Shared by keyevent/swipe/set_dnd -- their proposals differ only in how the command
    is built; none can carry a hidden extra effect once gated."""
    result = await device.run(command.command)
    return result.exit_code


async def take_screenshot(device: Device) -> bytes:
    """No candidates, no ambiguity, always safe (screencap is in
    gate.READ_ONLY_PREFIXES) -- same reasoning dumpsys uses to skip the gate
    entirely. exec-out streams the PNG on stdout; nothing is written to the device."""
    return await device.run_binary("screencap -p")


async def propose_dnd(jev: JudgeEngine, goal: str, *, verbose: bool = True) -> ClosedSetProposal:
    return await propose_from_closed_set(
        jev, goal, DND_MODES,
        options_key="mode_options",
        pick_instructions=question_sets.text("dnd.pick"),
        any_fit_instructions=question_sets.text("dnd.any_fit"),
        pick_verb="DND mode",
        command_for=lambda mode: CommandVariant(command=f"cmd notification set_dnd {mode}", rationale=f"set Do Not Disturb to {mode} per the goal"),
        label_for=lambda mode: f"set Do Not Disturb to {mode}",
        gate_instructions=DND_SAFE_INSTRUCTIONS,
        verbose=verbose, pick_qid="dnd.pick", any_fit_qid="dnd.any_fit", gate_qid="dnd.safe",
    )


@dataclass
class DumpsysOutcome:
    service: str | None
    confidence: float
    fit: float
    parsed: dict[str, str] | None = None
    answer_key: str | None = None
    reasons: tuple[str, ...] = ()
    # Journal linkage: the service pick's round-2 ground ask (or the answer-field
    # pick's when one ran), so the outcome row joins the decision row that chose it.
    call_id: str | None = None
    label_keys: tuple[str, ...] = ()  # the fit_i of whichever ask call_id carries
    gate_key: str | None = None  # ungated (accept_any_fitting skips the gate): always None


async def run_dumpsys_query(jev: JudgeEngine, device: Device, goal: str, *, verbose: bool = True) -> DumpsysOutcome:
    if verbose:
        print("--- step 2: real probe for the open argument's real enumeration ---")
    services_raw = await device.run("dumpsys -l")
    services = parse_dumpsys_services(services_raw.stdout)
    if verbose:
        print(f"parsed {len(services)} real dumpsys services (no model)\n")
        print("--- step 3: two-round semantic narrow over the real enumeration ---")

    verdict = await narrow_and_pick(
        jev, goal, services,
        instructions=question_sets.text("dumpsys_query.pick"),
        fit_instructions=question_sets.text("dumpsys_query.fit"),
        accept_any_fitting=True, pick_qid="dumpsys_query.pick", fit_qid="dumpsys_query.fit",
    )
    if verbose:
        print(f"shortlist: {verdict.shortlist}")
        print(f"picked service: {verdict.choice} (confidence {verdict.confidence:.2f}, fit {verdict.fit:.2f})\n")
    if not verdict.ok:
        if verbose:
            print(f"=== ESCALATED === {'; '.join(verdict.reasons)}")
        return DumpsysOutcome(None, verdict.confidence, verdict.fit, reasons=tuple(verdict.reasons), call_id=verdict.call_id,
                              label_keys=(verdict.fit_key,) if verdict.fit_key else ())

    if verbose:
        print(f"--- step 4: code builds the command deterministically: dumpsys {verdict.choice} ---")
    result = await device.run(f"dumpsys {verdict.choice}")
    parsed = parse_key_value(result.stdout)
    if verbose:
        print(f"parsed {len(parsed)} key:value pairs (no model)\n")

    answer_key = None
    field_verdict = None  # only assigned below when there's something to ask about -- an empty
    # parse must not raise, it just means no answer-field ask ever ran.
    if parsed:
        async def _field_values(shortlist: list[str]) -> dict[str, str]:
            return {c: parsed[c] for c in shortlist}

        # min_fit is the real bar for this read-only pick -- raw Choice confidence spreads
        # thin over 200+ plausible field names even for a clearly-correct answer. evidence_for
        # attaches real values only to the round-2 shortlist, not every round-1 chunk --
        # dumpsys settings alone can parse to 2000+ real fields.
        field_verdict = await narrow_and_pick(
            jev, goal, list(parsed),
            instructions=question_sets.text("dumpsys_field.pick"),
            fit_instructions=question_sets.text("dumpsys_field.fit"),
            evidence_for=_field_values,
            accept_any_fitting=True, pick_qid="dumpsys_field.pick", fit_qid="dumpsys_field.fit",
        )
        if field_verdict.ok:
            answer_key = field_verdict.choice
    if verbose:
        print("=== RESULT ===")
        for key, value in list(parsed.items())[:8]:
            print(f"  {key}: {value}")
        if answer_key:
            print(f"\nanswer: {answer_key} = {parsed[answer_key]}")
    field_asked_ok = field_verdict is not None and field_verdict.ok
    winning_verdict = field_verdict if field_asked_ok else verdict
    return DumpsysOutcome(verdict.choice, verdict.confidence, verdict.fit, parsed, answer_key,
                          call_id=field_verdict.call_id if field_asked_ok else verdict.call_id,
                          label_keys=(winning_verdict.fit_key,) if winning_verdict.fit_key else ())
