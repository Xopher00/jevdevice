"""SandboxDevice over a fake sandbox: the transport's run() contract matches
LocalShellTransport's, with no network and no Vercel SDK installed."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from jevdevice.device import (
    SANDBOX_DEVICE_NAME,
    CliDevice,
    Device,
    SandboxDevice,
    SandboxShellTransport,
)


class FakeSandbox:
    """Records run_process calls; answers from a queue of (stdout, stderr, returncode)."""

    name = "sbx-fake"

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def run_process(self, command, args=None, **kwargs):
        self.calls.append({"command": command, "args": list(args or []), **kwargs})
        stdout, stderr, code = self.replies.pop(0)
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=code)


def _device(*replies) -> tuple[SandboxDevice, FakeSandbox]:
    sandbox = FakeSandbox(*replies)
    return SandboxDevice(SandboxShellTransport(sandbox)), sandbox


def test_sandbox_device_is_a_device_named_apart_from_every_host() -> None:
    device, _ = _device()
    assert isinstance(device, Device)
    assert device.name == SANDBOX_DEVICE_NAME
    assert device.name != CliDevice().name


async def test_run_wraps_the_command_in_an_in_vm_timeout_and_returns_its_result() -> None:
    device, sandbox = _device(("hello\n", "", 0))
    result = await device.run("echo hello", timeout=7)
    assert (result.stdout, result.stderr, result.exit_code) == ("hello\n", "", 0)
    call = sandbox.calls[0]
    assert call["command"] == "timeout"
    assert call["args"] == ["--kill-after=2", "7", "bash", "-lc", "echo hello"]
    assert call["capture_output"] is True
    assert call["kill_after"] > 7


async def test_a_timeout_reads_exactly_like_the_local_transports() -> None:
    device, _ = _device(("partial", "", 124))
    result = await device.run("sleep 60", timeout=1)
    assert (result.stdout, result.stderr, result.exit_code) == ("", "timed out", 124)


async def test_failures_pass_through_with_their_exit_code_and_missing_output_is_empty() -> None:
    device, _ = _device((None, "no such file", 2))
    result = await device.run("cat /nope")
    assert (result.stdout, result.stderr, result.exit_code) == ("", "no such file", 2)


async def test_snapshot_member_runs_through_the_sandbox() -> None:
    device, sandbox = _device(("/vercel\nfile.txt\n", "", 0))
    snapshot = await device.dump_hierarchy()
    assert snapshot.startswith("$ pwd && ls -A\n")
    assert "file.txt" in snapshot
    assert sandbox.calls[0]["args"][-1] == "pwd && ls -A"


async def test_run_binary_round_trips_bytes_through_base64() -> None:
    payload = bytes(range(256))
    device, sandbox = _device((base64.b64encode(payload).decode(), "", 0))
    assert await device.run_binary("cat /bin/blob") == payload
    assert "base64 -w0" in sandbox.calls[0]["args"][-1]


async def test_run_binary_raises_on_failure_like_the_local_device() -> None:
    device, _ = _device(("", "boom", 1))
    with pytest.raises(RuntimeError, match="exit 1"):
        await device.run_binary("false")


async def test_describe_names_the_sandbox() -> None:
    assert await SandboxShellTransport(FakeSandbox()).describe() == "vercel-sandbox:sbx-fake"


async def test_window_size_is_fixed_not_the_hosts_terminal() -> None:
    device, _ = _device()
    assert await device.window_size() == (80, 24)


def test_the_vercel_sdk_is_imported_lazily_so_the_extra_stays_optional() -> None:
    import ast
    import inspect

    import jevdevice.device.sandbox as module

    top_level = [node for node in ast.parse(inspect.getsource(module)).body
                 if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [getattr(node, "module", None) or node.names[0].name for node in top_level]
    assert not any(name and name.startswith("vercel") for name in names)
