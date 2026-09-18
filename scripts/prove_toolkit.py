"""Prove the generalized toolkit shape on a goal that isn't app-launching:
"what is my battery level?" Jev picks the tool (closed set) and the grounded
service name (from a real enumeration); code owns every command string and
every bit of parsing. No small model needed for this case — the same
narrow-real-enumeration-then-Jev-picks pattern that worked for packages works
here too, because `dumpsys -l` is itself a real, enumerable candidate list.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prove_it import confidence_gate, fuzzy_narrow

from jevdevice.gate import GateVerdict, gate_command
from jevdevice.jev import Choice, JevClient, Noul
from jevdevice.gate import CommandVariant
from jevdevice.transport import AdbTransport

SERIAL = "192.168.100.11:46585"
GOAL = sys.argv[1] if len(sys.argv) > 1 else "what is my battery level?"

# Fixed, generic toolkit: not app-specific, applies to any goal on any Android device.
TOOLS = {
    "list_packages": "lists every installed app package",
    "dumpsys": "reads a live system service's status (battery, wifi, window focus, ...)",
    "getprop": "reads a static system/build property",
    "settings_get": "reads a stored system setting value",
    "toggle_service": "turns a device radio/service on or off (bluetooth, nfc, data — not wifi, see deny-list)",
}

# svc's controllable services are a small, genuinely fixed set (not app-specific).
TOGGLEABLE_SERVICES = {"bluetooth": None, "nfc": None, "data": None}


def parse_dumpsys_services(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().endswith(":")]


def parse_key_value(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.strip().partition(":")
        key, value = key.strip(), value.strip()
        if key and value and " " not in key:
            result[key] = value
    return result


async def run_toggle_service(jev: JevClient, transport: AdbTransport) -> None:
    print("--- step 2: Jev fills the two closed-set arguments (batched) ---")
    answers = await jev.ask(
        {"goal": GOAL},
        {
            "service": Choice(instructions="Which service does the goal refer to?", criteria=TOGGLEABLE_SERVICES),
            "enabled": Choice(instructions="Does the goal want it turned on or off?", criteria={"on": None, "off": None}),
        },
    )
    service_pick, enabled_pick = answers["service"], answers["enabled"]
    print(f"service: {service_pick.choice} (confidence {service_pick.confidence:.2f})")
    print(f"enabled: {enabled_pick.choice} (confidence {enabled_pick.confidence:.2f})\n")
    for pick in (service_pick, enabled_pick):
        ok, reason = confidence_gate(pick.probabilities, pick.confidence)
        if not ok:
            print(f"=== ESCALATED === {reason}")
            return

    verb = "enable" if enabled_pick.choice == "on" else "disable"
    command = CommandVariant(command=f"svc {service_pick.choice} {verb}", rationale=f"{verb} {service_pick.choice} per the goal")
    print(f"--- step 3: gate (deny-list + Jev Noul) before executing: {command.command!r} ---")
    gate_result = await gate_command(jev, command, chosen_label=f"{verb} {service_pick.choice}")
    print(f"gate verdict: {gate_result.verdict} ({gate_result.reason}, noul={gate_result.noul_confidence})\n")
    if gate_result.verdict != GateVerdict.APPROVED:
        print("=== NOT EXECUTED === gate did not approve")
        return

    print("--- step 4: real mutating command ---")
    result = await transport.run(command.command)
    print(f"exit_code={result.exit_code}")

    print("--- step 5: resolve the real dumpsys service name (svc and dumpsys use different namespaces) ---")
    services_raw = await transport.run("dumpsys -l")
    services = parse_dumpsys_services(services_raw.stdout)
    narrowed_services = fuzzy_narrow(service_pick.choice, services, limit=5)
    print(f"narrowed to: {narrowed_services}")
    resolve_answers = await jev.ask(
        {"goal": GOAL, "toggled_service": service_pick.choice, "candidate_dumpsys_services": narrowed_services},
        {"pick": Choice(instructions="Which dumpsys service would show this toggled service's real status "
                                       "(prefer the plain manager/status service over a vendor HAL/audio path)?",
                         criteria={s: None for s in narrowed_services})},
    )
    check_service = resolve_answers["pick"].choice
    print(f"checking real service: {check_service!r} (confidence {resolve_answers['pick'].confidence:.2f})\n")

    print("--- step 6: real check + Jev verify ---")
    check = await transport.run(f"dumpsys {check_service}")
    verify = await jev.ask(
        {"goal": GOAL, "service_state": check.stdout[:2000]},
        {"satisfied": Noul(instructions="Given service_state, is the goal now achieved?")},
    )
    satisfied = verify["satisfied"].noul
    print(f"\n=== RESULT ===\nJev verify noul: {satisfied:.2f}\ngoal met: {satisfied >= 0.5}")


async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    transport = AdbTransport(SERIAL)
    jev = JevClient(api_key)

    print(f"goal: {GOAL!r}\n")
    print("--- step 1: Jev picks the tool (closed set of 4, real Choice) ---")
    tool_answers = await jev.ask(
        {"goal": GOAL},
        {"tool": Choice(instructions="Which tool would answer or achieve this goal?", criteria=TOOLS)},
    )
    tool_pick = tool_answers["tool"]
    print(f"picked tool: {tool_pick.choice} (confidence {tool_pick.confidence:.2f})\n")
    ok, reason = confidence_gate(tool_pick.probabilities, tool_pick.confidence)
    if not ok:
        print(f"=== ESCALATED === {reason}")
        return

    if tool_pick.choice == "toggle_service":
        await run_toggle_service(jev, transport)
        return

    if tool_pick.choice != "dumpsys":
        print(f"(only dumpsys/toggle_service are wired up in this proof script; picked {tool_pick.choice!r}, stopping here)")
        return

    print("--- step 2: real probe for the open argument's real enumeration ---")
    services_raw = await transport.run("dumpsys -l")
    services = parse_dumpsys_services(services_raw.stdout)
    print(f"parsed {len(services)} real dumpsys services (no model)\n")

    print("--- step 3: generic fuzzy narrow + Jev picks the grounded service ---")
    narrowed = fuzzy_narrow(GOAL, services, limit=10)
    print(f"narrowed to: {narrowed}\n")
    assert all(s in services_raw.stdout for s in narrowed), "narrowed service not actually in real enumeration"

    service_answers = await jev.ask(
        {"goal": GOAL, "candidate_services": narrowed},
        {"service": Choice(instructions="Which service would answer this goal?", criteria={s: None for s in narrowed})},
    )
    service_pick = service_answers["service"]
    print(f"picked service: {service_pick.choice} (confidence {service_pick.confidence:.2f})\n")
    ok, reason = confidence_gate(service_pick.probabilities, service_pick.confidence)
    if not ok:
        print(f"=== ESCALATED === {reason}")
        return

    print(f"--- step 4: code builds the command deterministically: dumpsys {service_pick.choice} ---")
    result = await transport.run(f"dumpsys {service_pick.choice}")
    parsed = parse_key_value(result.stdout)
    print(f"parsed {len(parsed)} key:value pairs (no model)\n")

    print("=== RESULT ===")
    for key, value in list(parsed.items())[:8]:
        print(f"  {key}: {value}")
    if "level" in parsed:
        print(f"\nanswer: battery level is {parsed['level']}%")


if __name__ == "__main__":
    asyncio.run(main())
