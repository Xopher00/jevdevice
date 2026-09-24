"""Goal scope for journal rows, and the process-wide core `Journal` singleton.
Row shape, replay, and blob offload now live in typesymbolic; this module
keeps only what core lacks: goal context, jevdevice's env knobs, and
retention (core has no pruning).

Data lives OUTSIDE the repo (default ~/.jevdevice/tsjournal/,
JEV_JOURNAL_DIR overrides). The old ~/.jevdevice/journal/ dir is never
read, moved, or deleted -- this is a fresh start, not a migration.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from pathlib import Path

from typesymbolic.journal import Journal

DEFAULT_JOURNAL_DIR = Path.home() / ".jevdevice" / "tsjournal"
JOURNAL_MAX_AGE_DAYS = 0  # 0 = keep all history
FSYNC_EVERY_ROW = False

ENV_JOURNAL_DIR = "JEV_JOURNAL_DIR"
ENV_MAX_AGE_DAYS = "JEV_JOURNAL_MAX_AGE_DAYS"
ENV_ENABLED = "JEV_JOURNAL"  # "0"/"off" disables emission entirely (tests, benchmarks)

# (goal_id, goal_text) of the goal the current task is serving, set by public
# entry points via goal_scope(). Deep call sites (narrowing, gate) read it via
# current_goal() instead of threading goal_id through every signature.
_GOAL_CONTEXT: ContextVar[tuple[str, str] | None] = ContextVar("jevdevice_goal", default=None)


def goal_id_for(goal_text: str) -> str:
    """Stable short id so rows can be aggregated by goal across sessions."""
    return hashlib.sha256(goal_text.encode("utf-8")).hexdigest()[:12]


def current_goal() -> tuple[str | None, str | None]:
    ctx = _GOAL_CONTEXT.get()
    return (ctx[0], ctx[1]) if ctx else (None, None)


@contextmanager
def goal_scope(goal_text: str):
    """Public entry points (device_do, device_approve, run_toolkit) wrap their
    body in this so every ask() underneath fills the decision row's scope."""
    token = _GOAL_CONTEXT.set((goal_id_for(goal_text), goal_text))
    try:
        yield
    finally:
        _GOAL_CONTEXT.reset(token)


def journal_dir() -> Path:
    return Path(os.environ.get(ENV_JOURNAL_DIR) or DEFAULT_JOURNAL_DIR).expanduser()


def _prune_old_files(directory: Path, max_age_days: int) -> None:
    if max_age_days <= 0:
        return
    cutoff = datetime.now().date() - timedelta(days=max_age_days)  # noqa: DTZ005 -- naive, local retention sweep only
    for path in directory.glob("journal-*.jsonl"):
        try:
            file_date = datetime.strptime(path.stem.removeprefix("journal-"), "%Y%m%d").date()  # noqa: DTZ007
        except ValueError:
            continue
        if file_date < cutoff:
            path.unlink()


_default_journal: Journal | None = None


def get_journal() -> Journal:
    """Process-wide core Journal singleton, created lazily so env knobs set
    after import still apply (JEV_JOURNAL_DIR / JEV_JOURNAL_MAX_AGE_DAYS / JEV_JOURNAL)."""
    global _default_journal
    if _default_journal is None:
        disabled = os.environ.get(ENV_ENABLED, "1").strip().lower() in {"0", "off", "false", "no"}
        directory = journal_dir()
        journal = Journal(
            root=directory, enabled=not disabled, rotation="daily",
            background_writes=False, fsync=FSYNC_EVERY_ROW,
        )
        if journal.enabled:
            max_age_days = int(os.environ.get(ENV_MAX_AGE_DAYS, JOURNAL_MAX_AGE_DAYS))
            _prune_old_files(directory, max_age_days)
        _default_journal = journal
    return _default_journal
