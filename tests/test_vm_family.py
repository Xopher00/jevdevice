"""Offline checks for the seeded-VM family: frozen data shape (candidates,
CORRECT, EXPECT), split isolation (held-out never planned or given candidates),
SEED_SCRIPT has no network commands, and the one-sandbox re-seed helper's
ordering and cleanup -- all driven by fakes, no vercel SDK import."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PHASES = Path(__file__).resolve().parent.parent / "eval" / "phases"
if str(PHASES) not in sys.path:
    sys.path.insert(0, str(PHASES))

import vm_family
from splitguard import heldout_goals

# --- frozen data shape ------------------------------------------------------------


def test_dev_goals_have_frozen_candidates_correct_and_a_compiling_expect() -> None:
    for goal in vm_family.dev_vm_goals():
        gid = goal["id"]
        candidates = vm_family.CANDIDATE_COMMANDS[gid]
        assert 2 <= len(candidates) <= 4
        assert vm_family.CORRECT[gid] in candidates
        re.compile(vm_family.EXPECT[gid])


def test_heldout_goals_are_never_planned_and_carry_no_candidates() -> None:
    heldout_ids = {g["id"] for g in heldout_goals("vm_questions")}
    dev_ids = {g["id"] for g in vm_family.dev_vm_goals()}
    assert heldout_ids and heldout_ids.isdisjoint(dev_ids)
    for gid in heldout_ids:
        assert gid not in vm_family.CANDIDATE_COMMANDS
        assert gid not in vm_family.CORRECT
        assert gid not in vm_family.EXPECT


def test_seed_script_is_a_frozen_string_with_no_network_commands() -> None:
    assert isinstance(vm_family.SEED_SCRIPT, str) and vm_family.SEED_SCRIPT
    lowered = vm_family.SEED_SCRIPT.lower()
    for word in ("curl", "wget", "apt-get", "apt ", "pip install", "pip3 install", "nc -", "ssh "):
        assert word not in lowered


# --- one-sandbox re-seed helper: fakes only, no vercel import ----------------------


class FakeSandbox:
    """Records every run_process script; seeding fails when `seed_ok` is False."""

    def __init__(self, seed_ok: bool = True) -> None:
        self.seed_ok = seed_ok
        self.scripts: list[str] = []
        self.destroyed = False

    async def run_process(self, command, args=None, **kwargs):
        self.scripts.append(args[-1] if args else command)
        return SimpleNamespace(stdout="", stderr="" if self.seed_ok else "seed failed",
                               returncode=0 if self.seed_ok else 1)

    async def destroy(self) -> None:
        self.destroyed = True


class FakeClient:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox, self.created = sandbox, 0

    async def create_sandbox(self, **kwargs):
        self.created += 1
        return self.sandbox


async def _fake_new_sandbox(client):
    return await client.create_sandbox()


@pytest.fixture(autouse=True)
def _no_real_vercel_import(monkeypatch):
    monkeypatch.setattr(vm_family, "_new_sandbox", _fake_new_sandbox)


def _goals(n: int) -> list[dict]:
    return [{"id": f"g{i}"} for i in range(n)]


async def test_one_sandbox_per_run_reseeded_before_every_goal_then_destroyed() -> None:
    sandbox = FakeSandbox()
    client = FakeClient(sandbox)
    seen = []

    async def per_goal(goal, device):
        assert isinstance(device, vm_family.SandboxDevice)
        seen.append((goal["id"], len(sandbox.scripts)))
        return goal["id"]

    assert await vm_family.seed_and_run(client, _goals(3), per_goal) == ["g0", "g1", "g2"]
    assert client.created == 1
    assert sandbox.scripts == [vm_family.RESEED_SCRIPT] * 3
    assert seen == [("g0", 1), ("g1", 2), ("g2", 3)]  # each goal runs right after its own re-seed
    assert sandbox.destroyed


async def test_the_sandbox_is_destroyed_even_when_a_goal_raises() -> None:
    sandbox = FakeSandbox()

    async def per_goal(goal, device):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await vm_family.seed_and_run(FakeClient(sandbox), _goals(2), per_goal)
    assert sandbox.destroyed


async def test_a_failed_seed_raises_before_any_goal_and_still_destroys() -> None:
    sandbox = FakeSandbox(seed_ok=False)

    async def per_goal(goal, device):
        raise AssertionError("must not be reached")

    with pytest.raises(RuntimeError, match="seed script failed"):
        await vm_family.seed_and_run(FakeClient(sandbox), _goals(1), per_goal)
    assert sandbox.destroyed


def test_reseed_wipes_the_fixture_before_rebuilding_it() -> None:
    assert vm_family.RESEED_SCRIPT.startswith("rm -rf /tmp/vmfix && ")
    assert vm_family.RESEED_SCRIPT.endswith(vm_family.SEED_SCRIPT)
