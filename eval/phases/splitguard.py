"""The frozen dev/held-out split of eval/goals.yaml, as one shared loader.

Every batch/export/mining script asserts against the SAME split logic here
instead of each carrying its own copy -- the split is the project's sacred
boundary, and drift between private copies is exactly how contamination
happens. Held-out goals are never run, exported, tuned on, or trained on.

Importable both ways scripts are actually used: `uv run python
eval/phases/foo.py` (the script's directory is sys.path[0]) and from tests
that load a script by file location with this directory on sys.path.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from jevdevice.decision_log import goal_id_for

EVAL_DIR = Path(__file__).resolve().parent.parent
GOALS_FILE = EVAL_DIR / "goals.yaml"


def entries(family: str | None = None) -> list[dict]:
    """Flat {id, split, goal} entries, optionally one family (phone_questions...)."""
    data = yaml.safe_load(GOALS_FILE.read_text())
    families = [data[family]] if family else list(data.values())
    return [entry for entries in families for entry in entries]


def dev_goals(family: str | None = None) -> list[dict]:
    return [e for e in entries(family) if e.get("split") == "dev"]


def heldout_goals(family: str | None = None) -> list[dict]:
    return [e for e in entries(family) if e.get("split") == "heldout"]


def heldout_goal_ids(family: str | None = None) -> set[str]:
    return {goal_id_for(e["goal"]) for e in heldout_goals(family)}


def assert_dev_only(goals: list[dict]) -> None:
    """Hard guard for batch drivers: raise if any planned goal is held-out."""
    heldout = heldout_goal_ids()
    contaminated = sorted({goal_id_for(g["goal"]) for g in goals} & heldout)
    if contaminated:
        raise ValueError(f"held-out goals in the dev plan (split is sacred): {contaminated}")
