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
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import yaml
from typesymbolic.journal import OUTCOME
from typesymbolic.labels import VERDICT

from jevdevice.journal.decision_log import goal_id_for
from jevdevice.question_sets import load as load_question_set

DEFAULT_RECIPES_DIR = Path.home() / ".jevdevice" / "recipes"
ENV_RECIPES_DIR = "JEV_RECIPES_DIR"
RECIPES_FILE = "recipes.json"
BACKUP_FILE = "recipes.json.bak"

EMBED_DIM = 128          # hashed bag-of-tokens dimensionality
RETRIEVAL_FLOOR = 0.55   # min cosine similarity for a recipe hit (tunable on dev data)
MAX_CHAIN_GAP_S = 1800   # a chain is one run: rows farther apart than this are separate sessions
# These kinds now run a real post-action screen check (see dispatch.py); a row from
# before that change has only an exit code, so it can't be trusted as a recipe step.
CHECKED_KINDS = frozenset({"keyevent", "swipe", "set_dnd"})

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

def _final_chain(rows: list[dict], verdict_status: dict[str, str]) -> list[dict]:
    """The final run's rows: everything in the last session-window that closes
    with a verified outcome (`verdict_status`: call_id -> the joined verdict
    row's status). Rows farther apart than MAX_CHAIN_GAP_S are separate runs."""
    final = next((i for i in range(len(rows) - 1, -1, -1) if verdict_status.get(rows[i].get("call_id")) == "verified"), None)
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
    rows = list(rows if rows is not None else journal.replay())
    verdicts = (r for r in rows if r.get("type") == VERDICT and r.get("call_id"))
    verdict_status = {r["call_id"]: r.get("status") for r in verdicts}
    by_goal: dict[str, list[dict]] = {}
    goal_text: dict[str, str] = {}
    for row in rows:
        if row.get("type") != OUTCOME or not row.get("goal_id"):
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
                kind=row.get("kind"), goal=row.get("goal") or goal_text.get(goal_id, ""),
                command=row.get("executed_command"),
                from_node=(row.get("graph_edge") or {}).get("from_node"),
                to_node=(row.get("graph_edge") or {}).get("to_node"),
                call_id=row.get("call_id"),
            )
            for row in _final_chain(goal_rows, verdict_status)
            # device-verified steps only: dead ends (failed/escalated/unconfirmed
            # attempts) are the journey, not the chain
            if verdict_status.get(row.get("call_id")) == "verified"
            and row.get("kind")
            and (row.get("executed_command") or row.get("graph_edge"))
            # checked kinds need a real post-action satisfied verdict, not just exit code 0
            and (row.get("kind") not in CHECKED_KINDS or row.get("satisfied") is not None)
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
    must carry a verified Verdict row. `rows`: see recipes_from_journal."""
    verified_call_ids = {
        row.get("call_id")
        for row in (rows if rows is not None else journal.replay())
        if row.get("type") == VERDICT and row.get("status") == "verified"
    }
    return all(step.call_id in verified_call_ids for step in recipe.steps if step.call_id)


# --- rebuild ---------------------------------------------------------------------

@dataclass
class RebuildReport:
    n_recipes: int
    n_new: int              # goal_ids not already in the store before this rebuild
    skipped_heldout: int
    skipped_excluded: int
    dropped_unverified: int


def _rebuild_candidates(
    journal, *, heldout_goal_ids: frozenset[str], exclude: frozenset[str], rows: list[dict] | None,
) -> tuple[dict[str, Recipe], int, int, int]:
    """Filter-then-verify pass shared by `rebuild()` and the CLI's dry-run
    preview. Held-out/excluded goals are filtered from the rows before
    `recipes_from_journal` ever sees them -- filtering, not the raise that
    function does, because an interactive rebuild must not abort."""
    rows = list(rows if rows is not None else journal.replay())
    drop = frozenset(heldout_goal_ids) | frozenset(exclude)
    outcome_goal_ids = {row["goal_id"] for row in rows if row.get("type") == OUTCOME and row.get("goal_id")}
    skipped_heldout = len(outcome_goal_ids & heldout_goal_ids)
    skipped_excluded = len(outcome_goal_ids & frozenset(exclude))
    filtered_rows = [row for row in rows if row.get("type") != OUTCOME or row.get("goal_id") not in drop]
    candidates = recipes_from_journal(journal, rows=filtered_rows)
    verified = {goal_id: recipe for goal_id, recipe in candidates.items()
                if verify_recipe(recipe, journal, rows=filtered_rows)}
    dropped_unverified = len(candidates) - len(verified)
    return verified, skipped_heldout, skipped_excluded, dropped_unverified


def rebuild(
    store: RecipeStore, journal, *, heldout_goal_ids: frozenset[str] = frozenset(),
    exclude: frozenset[str] = frozenset(), rows: list[dict] | None = None, dry_run: bool = False,
) -> RebuildReport:
    """Rebuild the store from the current journal and REPLACE its contents
    (not merge): a recipe that no longer replays verified must not survive.
    `dry_run`: compute and report only, no backup, no write."""
    verified, skipped_heldout, skipped_excluded, dropped_unverified = _rebuild_candidates(
        journal, heldout_goal_ids=heldout_goal_ids, exclude=exclude, rows=rows,
    )
    n_new = len(set(verified) - {recipe.recipe_id for recipe in store.all()})
    if not dry_run:
        if store.path.exists():
            shutil.copy2(store.path, store.path.with_name(BACKUP_FILE))
        store.replace(verified)
    return RebuildReport(
        n_recipes=len(verified), n_new=n_new, skipped_heldout=skipped_heldout,
        skipped_excluded=skipped_excluded, dropped_unverified=dropped_unverified,
    )


def resolve_heldout_and_exclusions() -> tuple[frozenset[str], frozenset[str]]:
    """Held-out goal ids + operator exclusions, without importing eval/ into
    the package. Held-out replicates eval/phases/splitguard.heldout_goal_ids()
    exactly: eval/goals.yaml entries with split=="heldout", hashed with the
    same goal_id_for; exclusions come from build_card.json's
    "operator_exclusions" (eval/phases/build_recipes.py's own output), if it
    exists. Both are empty when eval/ isn't present in this checkout."""
    repo_root = Path(__file__).resolve().parents[3]
    heldout: set[str] = set()
    goals_file = repo_root / "eval" / "goals.yaml"
    if goals_file.exists():
        data = yaml.safe_load(goals_file.read_text()) or {}
        for family_entries in data.values():
            for entry in family_entries:
                if entry.get("split") == "heldout":
                    heldout.add(goal_id_for(entry["goal"]))
    exclude: set[str] = set()
    build_card = repo_root / "eval" / "phases" / "recipes" / "build_card.json"
    if build_card.exists():
        card = json.loads(build_card.read_text())
        exclude.update(card.get("operator_exclusions", []))
    return frozenset(heldout), frozenset(exclude)


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

    def replace(self, recipes: dict[str, Recipe]) -> None:
        """Wholesale replace (rebuild semantics), then save."""
        self._recipes = dict(recipes)
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
    """`python -m jevdevice.execution.recipes list` / `match "goal text"` /
    `rebuild [--dry-run]`. `eval/phases/build_recipes.py` remains the
    eval-provenance build (writes build_card.json); `rebuild` is the runtime
    demo/operator path over the same journal and split guard."""
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
    if args[0] == "rebuild":
        from jevdevice.journal.decision_log import get_journal

        dry_run = "--dry-run" in args[1:]
        heldout, exclude = resolve_heldout_and_exclusions()
        journal = get_journal()
        rows = list(journal.replay())  # one scan, shared by the preview and the report below
        candidates, *_ = _rebuild_candidates(journal, heldout_goal_ids=heldout, exclude=exclude, rows=rows)
        report = rebuild(store, journal, heldout_goal_ids=heldout, exclude=exclude, rows=rows, dry_run=dry_run)
        print(json.dumps(asdict(report), indent=2))
        print(json.dumps({
            "goals": sorted({recipe.goal for recipe in candidates.values()}),
            "kinds": sorted({step.kind for recipe in candidates.values() for step in recipe.steps}),
        }, indent=2))
        return 0
    print("usage: python -m jevdevice.execution.recipes [list] | match '<goal text>' | rebuild [--dry-run]",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
