"""Recipe store: goal-level verified chains aggregated from the
decision journal by goal_id.

A recipe is the verified step chain of one goal: only the steps that
reached a device-verified outcome, with dead-end and backtracking attempts
(failed tries, retried duplicates) removed before storing. Every step carries
its own goal text and kind, so replaying re-grounds each step's slots
(element, service, package) through the normal propose/narrow/gate path --
the recipe only fixes WHICH kinds run in WHICH order. A recipe is captured
from the journal, never invented.

Retrieval is by goal-text similarity: a deterministic hashed token-count
(unigrams + bigrams, sha256-bucketed, L2-normalized) of the goal text -- no
model, no network, reproducible across sessions. Cosine similarity >=
RETRIEVAL_FLOOR (a named knob, tunable on dev-half evidence only) counts as a
recipe hit.

Store location lives OUTSIDE the repo (like the journal): JEV_RECIPES_DIR or
~/.jevdevice/recipes/recipes.json.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from jevdevice.journal import decision_log
from jevdevice.question_sets import load as load_question_set

DEFAULT_RECIPES_DIR = Path.home() / ".jevdevice" / "recipes"
ENV_RECIPES_DIR = "JEV_RECIPES_DIR"
RECIPES_FILE = "recipes.json"

EMBED_DIM = 128          # hashed bag-of-tokens dimensionality
RETRIEVAL_FLOOR = 0.55   # min cosine similarity for a recipe hit (tunable on dev data)
MAX_CHAIN_GAP_S = 1800   # a chain is one run: rows farther apart than this are separate sessions

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


# --- goal-text embedding --------------------------------------------------------

def embed(text: str) -> list[float]:
    """Deterministic hashed bag-of-tokens embedding (unigrams + bigrams).
    Same text -> same vector, forever; no model is consulted."""
    vec = [0.0] * EMBED_DIM
    tokens = _TOKEN_RE.findall(text.lower())
    for feature in tokens + [f"{a} {b}" for a, b in pairwise(tokens)]:
        digest = int.from_bytes(hashlib.sha256(feature.encode("utf-8")).digest()[:8], "big")
        vec[digest % EMBED_DIM] += 1.0 if (digest >> 63) & 1 else -1.0
    norm = math.sqrt(sum(component * component for component in vec))
    return [component / norm for component in vec] if norm else vec


def similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# --- recipe shape -------------------------------------------------------------

@dataclass
class RecipeStep:
    kind: str           # one of dispatch.ACTION_KINDS
    goal: str           # the step's OWN goal text (its slots re-ground on replay)
    command: str | None # the executed command, if one ran
    from_node: str | None
    to_node: str | None
    call_id: str | None # journal linkage: joins back to the decision rows that produced it


@dataclass
class Recipe:
    recipe_id: str      # == goal_id (sha256(goal)[:12], stable across sessions)
    goal: str           # the top-level goal text the chain served
    goal_id: str
    steps: list[RecipeStep]
    embedding: list[float]
    created: str        # ts of the final verified outcome this chain was captured from
    question_set: str   # the frozen question-set version the original run used

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Recipe:
        data = dict(data)
        data["steps"] = [RecipeStep(**step) for step in data["steps"]]
        return cls(**data)


# --- journal -> recipes (dead-end/backtracking removal) ------------------------

def _final_chain(rows: list[dict]) -> list[dict]:
    """The final run's rows: everything in the last session-window that closes
    with a verified outcome. Rows farther apart than MAX_CHAIN_GAP_S are
    separate runs, so an old session's chain never pollutes today's; within the
    window, failed/escalated rows are the journey (dropped downstream) and only
    the verified steps can enter the chain."""
    final = next((i for i in range(len(rows) - 1, -1, -1) if rows[i].get("verification") == decision_log.VERIFIED), None)
    if final is None:
        return []
    start = 0
    final_ts = _parse_ts(rows[final].get("ts", ""))
    for i in range(final - 1, -1, -1):
        ts = _parse_ts(rows[i].get("ts", ""))
        if final_ts is not None and ts is not None and (final_ts - ts).total_seconds() > MAX_CHAIN_GAP_S:
            start = i + 1
            break
    return rows[start:final + 1]


def _compress(steps: list[RecipeStep]) -> list[RecipeStep]:
    """A repeated (kind, command) compresses to its LAST occurrence (the
    earlier one was a backtrack), keeping last-occurrence order. Only
    verified rows enter the chain, so dead ends never reach here."""
    kept: list[RecipeStep] = []
    for i, step in enumerate(steps):
        if any(later.kind == step.kind and later.command == step.command for later in steps[i + 1:]):
            continue  # a later identical step supersedes this backtracked one
        kept.append(step)
    return kept


def recipes_from_journal(
    journal, *, heldout_goal_ids: frozenset[str] = frozenset(), rows: list[dict] | None = None,
) -> dict[str, Recipe]:
    """Every derivable recipe, keyed by goal_id. Outcome rows carry what a
    step needs (kind, graph_edge, command, call_id); decision rows are not
    read. Raises ValueError if a held-out goal would enter the store: a
    recipe is training data and the dev/held-out split must hold.
    `rows`: pass an already-materialized `journal.replay()` to skip a second
    full replay when the caller also calls verify_recipe right after."""
    by_goal: dict[str, list[dict]] = {}
    goal_text: dict[str, str] = {}
    for row in rows if rows is not None else journal.replay():
        if row.get("type") != decision_log.OUTCOME or not row.get("goal_id"):
            continue
        goal_id = row["goal_id"]
        by_goal.setdefault(goal_id, []).append(row)
        goal_text.setdefault(goal_id, row.get("goal") or "")
    contaminated = sorted(set(by_goal) & set(heldout_goal_ids))
    if contaminated:
        raise ValueError(f"held-out goals found in journal, refusing to build recipes: {contaminated}")
    question_set = load_question_set().version
    recipes: dict[str, Recipe] = {}
    for goal_id, goal_rows in by_goal.items():
        chain = [
            RecipeStep(
                kind=row["kind"], goal=row.get("goal") or goal_text.get(goal_id, ""),
                command=row.get("executed_command"),
                from_node=(row.get("graph_edge") or {}).get("from_node"),
                to_node=(row.get("graph_edge") or {}).get("to_node"),
                call_id=row.get("call_id"),
            )
            for row in _final_chain(goal_rows)
            # device-verified steps only: dead ends (failed/escalated/unconfirmed
            # attempts) are the journey, not the chain
            if row.get("verification") == decision_log.VERIFIED
            and row.get("kind")
            and (row.get("executed_command") or row.get("graph_edge"))
        ]
        chain = [step for step in _compress(chain) if step.kind]
        if not chain:
            continue
        goal = goal_text.get(goal_id) or chain[-1].goal
        recipes[goal_id] = Recipe(
            recipe_id=goal_id, goal=goal, goal_id=goal_id, steps=chain,
            embedding=embed(goal), created=goal_rows[-1].get("ts", ""),
            question_set=question_set,
        )
    return recipes


def verify_recipe(recipe: Recipe, journal, *, rows: list[dict] | None = None) -> bool:
    """A stored recipe replays to its verified outcome: every step's call_id
    must exist in the journal as a VERIFIED outcome row (missing/failed steps
    mean the recipe drifted from device truth and must be re-captured).
    `rows`: see recipes_from_journal -- pass the same materialized replay to
    avoid a second full journal scan."""
    verified_call_ids = {
        row.get("call_id")
        for row in (rows if rows is not None else journal.replay())
        if row.get("type") == decision_log.OUTCOME and row.get("verification") == decision_log.VERIFIED
    }
    return all(step.call_id in verified_call_ids for step in recipe.steps if step.call_id)


# --- store ---------------------------------------------------------------------

def _store_path(directory: Path | str | None = None) -> Path:
    env_dir = os.environ.get(ENV_RECIPES_DIR)
    root = Path(directory or env_dir or DEFAULT_RECIPES_DIR).expanduser()
    return root / RECIPES_FILE


class RecipeStore:
    """JSON-file store of recipes keyed by goal_id, retrieved by goal-text
    File format versioned implicitly by the Recipe fields; unknown/extra fields
    raise on load (fail-closed, like the question sets)."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self.path = _store_path(directory)
        self._recipes: dict[str, Recipe] = {}
        if self.path.exists():
            for goal_id, data in json.loads(self.path.read_text()).items():
                self._recipes[goal_id] = Recipe.from_dict(data)

    def all(self) -> list[Recipe]:
        return list(self._recipes.values())

    def get(self, goal_id: str) -> Recipe | None:
        return self._recipes.get(goal_id)

    def upsert(self, recipe: Recipe) -> None:
        self._recipes[recipe.recipe_id] = recipe
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {goal_id: recipe.to_dict() for goal_id, recipe in self._recipes.items()}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def best_match(self, goal_text: str, floor: float = RETRIEVAL_FLOOR) -> tuple[Recipe, float] | None:
        """Nearest recipe by goal-text cosine similarity, above the floor."""
        query = embed(goal_text)
        best: tuple[float, Recipe] | None = None
        for recipe in self._recipes.values():
            score = similarity(query, recipe.embedding)
            if best is None or score > best[0]:
                best = (score, recipe)
        if best is None or best[0] < floor:
            return None
        return best[1], best[0]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    """`python -m jevdevice.execution.recipes list` / `match "goal text"`. Building from
    the journal is an eval-script job (eval/phases/build_recipes.py) because
    the dev/held-out split guard needs eval/goals.yaml."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    store = RecipeStore()
    if not args or args[0] == "list":
        for recipe in store.all():
            print(json.dumps({"recipe_id": recipe.recipe_id, "goal": recipe.goal, "steps": len(recipe.steps)}))
        return 0
    if args[0] == "match" and len(args) > 1:
        hit = store.best_match(args[1])
        print(json.dumps({"matched": hit is not None, "score": None if hit is None else round(hit[1], 3),
                          "recipe_id": None if hit is None else hit[0].recipe_id}))
        return 0
    print("usage: python -m jevdevice.execution.recipes [list] | match '<goal text>'", file=sys.stderr)
    return 2
