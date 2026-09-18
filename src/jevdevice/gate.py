"""Gate: every mutating command passes a deny-list, then a Jev Noul, before it runs.

Classification never trusts what the small model calls the command — is_read_only/
is_denied are applied to the literal command string regardless of any label the
model attached.
"""

from __future__ import annotations

from dataclasses import dataclass

from .grounding import is_denied, is_read_only
from .jev import JevClient, Noul


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


async def gate_command(jev: JevClient, command: CommandVariant, chosen_label: str, threshold: float = 0.8) -> GateResult:
    if is_read_only(command.command):
        return GateResult(verdict=GateVerdict.APPROVED, reason="read_only")
    if is_denied(command.command):
        return GateResult(verdict=GateVerdict.DENIED, reason="deny_listed")

    answers = await jev.ask(
        {"chosen_action": chosen_label, "proposed_command": command.command, "rationale": command.rationale},
        {"safe": Noul(
            instructions="Does the proposed_command's target and effect match the chosen_action "
                         "(same service/component, same on-or-off direction)? Answer no if it names a "
                         "different target, a different effect, or chains on any additional command.",
        )},
    )
    confidence = answers["safe"].noul
    if confidence >= threshold:
        return GateResult(verdict=GateVerdict.APPROVED, reason="jev_confirmed", noul_confidence=confidence)
    return GateResult(verdict=GateVerdict.NEEDS_APPROVAL, reason="jev_uncertain", noul_confidence=confidence)
