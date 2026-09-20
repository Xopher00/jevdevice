"""Per-engine budget profile: every option-count bound, state-size cut, and
screen truncation is a named knob on one profile per engine -- never a bare
int at a call site.

Why one profile per engine: the hosted engine caps a Choice at 255 options
with a ~32k-token state budget, so the jev profile keeps those historical
values and behavior is unchanged. The in-process engine's head is a TOKEN
budget (head_max_len=256 fits ~20 short options; the option markers alone
must fit) with a total max_len=1024 where overflow raises -- so its profile
bounds everything the judge sees accordingly.

Engine-name constants live here (not common.py) so pure modules -- matching,
narrowing, elements -- can read the profile without importing the client
wiring (import cycle: common -> jev -> decision_log; budget imports neither).

`current_profile(engine)` keys off the ANSWERING ENGINE, not the process env,
so both engines can run in one process without fighting over one set of
knobs. With no argument it falls back to JEV_ENGINE, the same env bootstrap()
reads, for engine-less call sites (describe_screen).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ENGINE_ENV = "JEV_ENGINE"
JEV_ENGINE_NAME = "jev"
LAYA_ENGINE_NAME = "laya"
ENGINES = (JEV_ENGINE_NAME, LAYA_ENGINE_NAME)

# The abstain option: every choice question that offers it also offers this,
# so "nothing here fits" is expressible as a pick instead of a forced guess.
# Picking it must escalate -- fail-closed -- never act.
NONE_OF_THESE = "none_of_these"
NONE_OF_THESE_CRITERION = (
    "None of the other options satisfies the goal -- pick this to abstain rather than guess."
)


@dataclass(frozen=True)
class BudgetProfile:
    """The named knobs one engine's budget is expressed through."""

    engine: str
    max_len: int            # total state token budget
    head_max_len: int       # head token budget (how many short options a Choice can carry)
    chunk_size: int         # chunk_candidates bound (round-1 recall + the fused-pick threshold)
    screen_limit: int       # describe_screen label cap (state text)
    shortlist_size: int     # retrieval shortlist cap before ONE choice call
    shortlist_first: bool   # try the retrieval shortlist before the chunked sweep
    short_labels: bool      # element options are short raw labels (rich descriptions stay in state text)
    descriptions_in_state: bool  # Choice criteria values carry the per-option rich description
    abstain_option: bool    # offer "none_of_these" in every choice
    probe_max_chars: int    # raw probe text (dumpsys status) clipped into state, chars


# The hosted engine keeps today's values exactly: 200-candidate chunks as
# headroom under the 255-option cap, 150-label screens, no retrieval
# shortlist, no abstain option -- the world this code was calibrated in.
JEV_PROFILE = BudgetProfile(
    engine=JEV_ENGINE_NAME,
    max_len=32_000,
    head_max_len=255,
    chunk_size=200,
    screen_limit=150,
    shortlist_size=200,
    shortlist_first=False,
    short_labels=False,
    descriptions_in_state=False,
    abstain_option=False,
    probe_max_chars=2000,
)

# The in-process engine: 20-candidate chunks under a 256-token head, 18-label
# screens, and every high-cardinality decision shortlist-shaped (scored top-20
# -> ONE choice call, chunking only as fallback). probe_max_chars: a raw
# dumpsys status rides in state under max_len=1024 -- the hosted engine's
# 2000-char clip alone would risk the budget there.
LAYA_PROFILE = BudgetProfile(
    engine=LAYA_ENGINE_NAME,
    max_len=1024,
    head_max_len=256,
    chunk_size=20,
    screen_limit=18,
    shortlist_size=20,
    shortlist_first=True,
    short_labels=True,
    descriptions_in_state=True,
    abstain_option=True,
    probe_max_chars=1200,
)

PROFILES: dict[str, BudgetProfile] = {JEV_ENGINE_NAME: JEV_PROFILE, LAYA_ENGINE_NAME: LAYA_PROFILE}


def profile_for(engine: str) -> BudgetProfile:
    """The budget profile of one answering engine. Unknown names fall back to
    the jev profile rather than guess: a test double without an engine_name
    gets the historical (jev) knobs, exactly the behavior it was written under."""
    return PROFILES.get(engine, JEV_PROFILE)


def current_profile(engine: str | None = None) -> BudgetProfile:
    """Profile of the given engine, else of the process engine (JEV_ENGINE --
    the same env bootstrap() reads; no client handle at the call site)."""
    if engine is None:
        engine = (os.environ.get(ENGINE_ENV) or JEV_ENGINE_NAME).strip().lower()
    return profile_for(engine)


def choice_criteria(options: dict[str, str | None], profile: BudgetProfile) -> dict[str, str | None]:
    """The option map a judge Choice actually sees: the real options plus, on
    profiles that offer it, the abstain option -- an explicit "none of these"
    the judge can pick instead of being forced to guess. A pick of none_of_these
    must escalate (see is_abstain); if a real option is genuinely named
    none_of_these, the real option wins and the abstain is simply not offered."""
    if not profile.abstain_option or NONE_OF_THESE in options:
        return dict(options)
    return {**options, NONE_OF_THESE: NONE_OF_THESE_CRITERION}


def is_abstain(choice: str | None) -> bool:
    """Did the judge abstain? Picking none_of_these escalates, never guesses."""
    return choice == NONE_OF_THESE
