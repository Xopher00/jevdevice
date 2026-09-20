"""Decision journal: append-only JSONL capture of every judge decision and its
downstream outcome, joined by call_id.

ledger.py counts tokens -- it can't train anything. The full decision context
(state snapshot, candidate set, full answer distribution) evaporates when the
screen changes; this journal is the raw material for recalibration, engine
A/B, fine-tune export and trajectory mining. Two invariants:

- REPLAY: reading the journal reconstructs exactly what ask() sent and got --
  blob refs resolve back to the original values on read (replay()).
- BIG VALUES OUT-OF-LINE: raw dumps/screenshots/huge candidate lists go to a
  content-addressed store (sha256 -> file under the journal data dir); rows
  keep refs + hashes only, which also keeps each JSONL line small enough that
  concurrent writers can't realistically tear one another's lines.

Fail-open: a journal write failure must never break a live decision --
record_* catch, print, and return. Data lives OUTSIDE the repo
(default ~/.jevdevice/journal/, JEV_JOURNAL_DIR overrides).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from pathlib import Path

# --- knobs (named constants; directory/age/enabled also overridable via env) ---
DEFAULT_JOURNAL_DIR = Path.home() / ".jevdevice" / "journal"
BLOB_MIN_BYTES = 2048  # values serialized larger than this are stored out-of-line
FSYNC_EVERY_ROW = False  # journaling is best-effort: a crash may lose the tail
JOURNAL_MAX_AGE_DAYS = 0  # 0 = keep all history

ENV_JOURNAL_DIR = "JEV_JOURNAL_DIR"
ENV_MAX_AGE_DAYS = "JEV_JOURNAL_MAX_AGE_DAYS"
ENV_ENABLED = "JEV_JOURNAL"  # "0"/"off" disables emission entirely (tests, benchmarks)

DECISION = "decision"
OUTCOME = "outcome"
CALIBRATION = "calibration"
UNLABELED = "unlabeled"

# Verification values for outcome rows. "verified" = the flow's own check passed
# (status ok); "failed" = ran and provably failed (non-zero exit); "escalated" =
# no execution, awaiting human resolution; "none" = ran but nothing confirmed it.
VERIFIED = "verified"
FAILED = "failed"
ESCALATED = "escalated"
NONE = "none"

# (goal_id, goal_text) of the goal the current task is serving, set by public
# entry points via goal_scope(). ask() reads it so deep call sites (narrowing,
# gate) don't need goal_id threaded through every signature.
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
    body in this so every ask() underneath is journaled with the goal it serves."""
    token = _GOAL_CONTEXT.set((goal_id_for(goal_text), goal_text))
    try:
        yield
    finally:
        _GOAL_CONTEXT.reset(token)


def _json_bytes(value) -> bytes:
    """Canonical serialization for hashing/list-walking -- same shape the JSONL
    writer uses, so a replayed row re-serializes to the identical bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


class BlobStore:
    """Content-addressed store under the journal dir: sha256 -> blob file."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, content: bytes) -> dict:
        sha256 = hashlib.sha256(content).hexdigest()
        path = self.root / sha256[:2] / sha256
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
            tmp.write_bytes(content)
            os.replace(tmp, path)
        return {"sha256": sha256, "bytes": len(content)}

    def get(self, ref: dict) -> bytes | None:
        sha256 = ref.get("sha256", "")
        if not sha256:
            return None
        path = self.root / sha256[:2] / sha256
        return path.read_bytes() if path.exists() else None


class DecisionJournal:
    """Append-only JSONL, one file per day (journal-YYYYMMDD.jsonl). Writes are
    a synchronous append per row -- no per-row fsync unless `fsync` says so --
    which keeps the hot-loop cost in the microseconds.
    asyncio-single-threaded: one writer per event loop, like UsageLedger."""

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        enabled: bool = True,
        max_age_days: int | None = None,
        fsync: bool = FSYNC_EVERY_ROW,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.enabled = enabled
        self.fsync = fsync
        self.clock = clock
        env_dir = os.environ.get(ENV_JOURNAL_DIR)
        self.directory = Path(directory or env_dir or DEFAULT_JOURNAL_DIR).expanduser()
        self.max_age_days = JOURNAL_MAX_AGE_DAYS if max_age_days is None else max_age_days
        # env only fills the knob when the caller left it unset
        if max_age_days is None and (env_age := os.environ.get(ENV_MAX_AGE_DAYS)):
            self.max_age_days = int(env_age)
        self.blobs = BlobStore(self.directory / "blobs")
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._prune_old_files()

    # -- writing ---------------------------------------------------------

    def _day_file(self) -> Path:
        return self.directory / f"journal-{self.clock().strftime('%Y%m%d')}.jsonl"

    def _append(self, row: dict) -> None:
        line = json.dumps(row, separators=(",", ":"), default=str)
        with open(self._day_file(), "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            if self.fsync:
                handle.flush()
                os.fsync(handle.fileno())

    def _prune_old_files(self) -> None:
        if self.max_age_days <= 0:
            return
        cutoff = self.clock().date() - timedelta(days=self.max_age_days)
        for path in self.directory.glob("journal-*.jsonl"):
            try:
                file_date = datetime.strptime(path.stem.removeprefix("journal-"), "%Y%m%d").date()  # noqa: DTZ007 -- naive dates only, same clock family as writes
            except ValueError:
                continue
            if file_date < cutoff:
                path.unlink()

    def _blobify(self, value):
        """Swap any value whose serialized form exceeds BLOB_MIN_BYTES for a
        sha256 ref. Recurses into children FIRST so one huge field doesn't drag
        its small siblings out-of-line; a container still too big after that
        (many medium values) is atomized whole. replay() reverses this exactly."""
        if isinstance(value, dict):
            value = {key: self._blobify(item) for key, item in value.items()}
        elif isinstance(value, list):
            value = [self._blobify(item) for item in value]
        elif not isinstance(value, str):
            return value
        encoded = _json_bytes(value) if not isinstance(value, str) else value.encode("utf-8")
        if len(encoded) <= BLOB_MIN_BYTES:
            return value
        encoding = "utf8" if isinstance(value, str) else "json"
        return {"blob": {**self.blobs.put(encoded), "encoding": encoding}}

    def record_decision(
        self, *, call_id: str, engine: str, model_revision: str, phase: str | None = None,
        state=None, questions=None, answers=None, truncation=None, usage=None, error=None,
        goal: str | None = None, goal_id: str | None = None,
        elapsed_ms: float | None = None, shadow_of: str | None = None,
    ) -> None:
        """One row per ask(): the full replayable decision. `answers` is the FULL
        distribution (probabilities/confidence/noul), not just the winning pick;
        on failure `answers` is None and `error` carries the message.
        elapsed_ms (P5): this ask()'s wall time. shadow_of (P5): set only on
        shadow rows -- the primary row's call_id this observation shadows; a
        None value means the row IS a primary decision (P4.5/analyses key off
        this, so a shadow row can never be mistaken for a real one)."""
        if not self.enabled:
            return
        row = {
            "type": DECISION,
            "ts": self.clock().isoformat(timespec="milliseconds"),
            "call_id": call_id,
            "goal_id": goal_id,
            "goal": goal,
            "engine": engine,
            "model_revision": model_revision,
            "phase": phase or UNLABELED,
            "state": state,
            "questions": questions,
            "answers": answers,
            "truncation": truncation,
            "usage": usage,
            "error": error,
            "elapsed_ms": elapsed_ms,
            "shadow_of": shadow_of,
        }
        try:
            self._append({key: self._blobify(value) for key, value in row.items()})
        except Exception as exc:  # noqa: BLE001 -- fail-open: telemetry must never break the decision path
            print(f"journal write failed: {exc}")

    def record_outcome(
        self, *, call_id: str | None = None, executed_command: str | None = None,
        verification: str = NONE, status: str | None = None,
        recovery_command: str | None = None, graph_edge: dict | None = None,
        device: str | None = None, decision: str | None = None,
        reasons=None, exit_code: int | None = None, satisfied: float | None = None,
        goal: str | None = None, goal_id: str | None = None,
    ) -> None:
        """One row per execution-flow event (ran / needs approval / denied),
        joined to its decision row(s) by call_id."""
        if not self.enabled:
            return
        row = {
            "type": OUTCOME,
            "ts": self.clock().isoformat(timespec="milliseconds"),
            "call_id": call_id,
            "goal_id": goal_id,
            "goal": goal,
            "device": device,
            "executed_command": executed_command,
            "verification": verification,
            "status": status,
            "recovery_command": recovery_command,
            "graph_edge": graph_edge,
            "decision": decision,
            "reasons": reasons,
            "exit_code": exit_code,
            "satisfied": satisfied,
        }
        try:
            self._append({key: self._blobify(value) for key, value in row.items()})
        except Exception as exc:  # noqa: BLE001 -- fail-open: telemetry must never break the action path
            print(f"journal write failed: {exc}")

    # -- reading ---------------------------------------------------------

    def record_calibration(
        self, *, event: str, engine: str | None = None, provenance: dict | None = None,
        result: dict | None = None,
    ) -> None:
        """One row per continuous-calibration loop run (P4.5): the proposed
        fits (or the documented non-proposal), the shadow-run verdicts under
        current vs proposed thresholds, and the window provenance. Additive
        row type -- replay() yields it like any other row."""
        if not self.enabled:
            return
        row = {
            "type": CALIBRATION,
            "ts": self.clock().isoformat(timespec="milliseconds"),
            "event": event,
            "engine": engine,
            "provenance": provenance,
            "result": result,
        }
        try:
            self._append({key: self._blobify(value) for key, value in row.items()})
        except Exception as exc:  # noqa: BLE001 -- fail-open: telemetry must never break the decision path
            print(f"journal write failed: {exc}")

    def _resolve_value(self, value):
        if (
            isinstance(value, dict) and len(value) == 1
            and isinstance(value.get("blob"), dict) and "sha256" in value["blob"]
        ):
            ref = value["blob"]
            content = self.blobs.get(ref)
            if content is None:
                return value  # missing blob: keep the ref, fail-open on read
            if ref.get("encoding") == "json":
                return json.loads(content)
            return content.decode("utf-8")
        if isinstance(value, dict):
            return {key: self._resolve_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve_value(item) for item in value]
        return value

    def replay(self, day: str | None = None) -> Iterator[dict]:
        """Every row in write order across day files, blob refs resolved -- this
        reconstructs exactly what ask() sent and got (crash-resume, cross-engine
        re-runs). Torn/corrupt lines are skipped, not fatal."""
        if not self.enabled:
            return
        pattern = f"journal-{day}.jsonl" if day else "journal-*.jsonl"
        for path in sorted(self.directory.glob(pattern)):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield self._resolve_value(row)


_default_journal: DecisionJournal | None = None


def get_journal() -> DecisionJournal:
    """Module-level singleton, created lazily so env knobs set after import
    still apply (JEV_JOURNAL_DIR / JEV_JOURNAL_MAX_AGE_DAYS / JEV_JOURNAL)."""
    global _default_journal
    if _default_journal is None:
        disabled = os.environ.get(ENV_ENABLED, "1").strip().lower() in {"0", "off", "false", "no"}
        _default_journal = DecisionJournal(enabled=not disabled)
    return _default_journal
