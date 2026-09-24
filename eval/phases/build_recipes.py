"""Build the recipe store from the journal -- journal-only, zero model calls,
zero device. The dev/held-out split is asserted hard: a held-out goal text
appearing anywhere in the journal (or the built store) aborts the build with a
non-zero exit, because a recipe is replay/training data.

Output: the recipe store (JEV_RECIPES_DIR or ~/.jevdevice/recipes/recipes.json)
+ eval/phases/recipes/build_card.json (counts + provenance).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


from jevdevice.execution.recipes import RecipeStore, recipes_from_journal, verify_recipe
from jevdevice.journal.decision_log import get_journal

OUT_DIR = REPO / "eval" / "phases" / "recipes"


import splitguard


def main() -> int:
    args = sys.argv[1:]
    # Operator-only exclusions (human sign-off, recorded in the card): a
    # held-out goal id whose rows pre-date this guard (a stray early run).
    # Excluding is an operator decision made by passing the flag here --
    # never a silent default.
    exclude: set[str] = set()
    i = 0
    while i < len(args):
        if args[i] == "--exclude" and i + 1 < len(args):
            exclude.add(args[i + 1])
            i += 2
        else:
            print(f"unknown argument {args[i]!r}", file=sys.stderr)
            return 2
    heldout = splitguard.heldout_goal_ids()
    journal = get_journal()
    rows = journal.replay()  # one full journal scan, shared with verify_recipe below
    # Fail-closed on contamination: recipes_from_journal raises on any held-out
    # goal unless the operator explicitly excluded it above (card-recorded).
    built = recipes_from_journal(journal, heldout_goal_ids=frozenset(heldout - exclude), rows=rows)
    store = RecipeStore()
    for recipe in built.values():
        store.upsert(recipe)
    # every stored recipe must replay to its verified outcome
    unreplayable = [r.recipe_id for r in store.all() if not verify_recipe(r, journal, rows=rows)]
    store_goal_ids = {r.recipe_id for r in store.all()}
    assert not (store_goal_ids & heldout), "held-out goal leaked into the recipe store"
    card = {
        "recipes_total": len(store.all()),
        "built_this_run": len(built),
        "unreplayable": unreplayable,
        "heldout_goal_ids_checked": len(heldout),
        "operator_exclusions": sorted(exclude),
        "steps_per_recipe": {r.recipe_id: len(r.steps) for r in store.all()},
        "kinds": sorted({step.kind for r in store.all() for step in r.steps}),
        "store": str(store.path),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "build_card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2))
    return 1 if unreplayable else 0


if __name__ == "__main__":
    raise SystemExit(main())
