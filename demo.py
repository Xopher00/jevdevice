"""Interactive terminal demo: connect an Android phone over wireless adb, then run
ready-made or free-typed plain-language goals through the real jevdevice pipeline
(pick action kind -> propose from live device state -> safety gate -> execute -> verify).

    python3 demo.py        (or: uv run python demo.py)
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


def _ensure_project_env() -> None:
    """Re-exec under the project's .venv when launched with a system python that can't import jevdevice."""
    try:
        import rich  # noqa: F401

        import jevdevice  # noqa: F401
        return
    except ImportError:
        pass
    root = Path(__file__).resolve().parent
    venv_python = root / ".venv" / "bin" / "python"
    if not venv_python.exists() and shutil.which("uv"):
        print("Setting up the project environment (one-time, `uv sync`)…", flush=True)
        subprocess.run(["uv", "sync", "--quiet"], cwd=root, check=False)
    if not venv_python.exists():
        sys.exit("jevdevice isn't installed. From this folder run:  uv sync   then:  python3 demo.py")
    if Path(sys.prefix).resolve() == (root / ".venv").resolve():
        sys.exit("The project environment is incomplete. From this folder run:  uv sync")
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    _ensure_project_env()

# Only importable once _ensure_project_env has run.
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Bound to the real stdout: pipeline calls run under redirect_stdout, and prompts
# and spinners must still reach the terminal while they do.
console = Console(file=sys.stdout, highlight=False)
ACCENT = "cyan"
SCREENSHOT_DIR = Path(__file__).resolve().parent / "demo_screenshots"

TOUR = [
    ("what is my battery level?", "Reads a system service. Nothing on the phone changes."),
    ("open the calculator", "Finds the installed calculator app and launches it."),
    ("tap the 7 button", "Taps a real on-screen button. Watch the safety gate."),
    ("go back to the home screen", "Presses the Home key."),
    ("take a screenshot", f"Saves the current screen to {SCREENSHOT_DIR.name}/."),
]

EXAMPLES = [
    "what wifi network am I connected to?",
    "open settings",
    "scroll down",
    "turn bluetooth off",
    "set do not disturb to priority only",
]

# Kinds whose response carries only the command's exit code: no post-action check runs.
EXIT_CODE_ONLY = {"keyevent", "swipe", "set_dnd"}

PHRASING = {
    "open_app": "open <app name>",
    "dumpsys": "what is my <battery level / wifi network>?",
    "toggle_service": "turn bluetooth on",
    "tap": "tap the <button name> button",
    "long_press": "long-press <item>",
    "type_text": "type <text> into the <field> field",
    "keyevent": "press the <home / back / camera> key",
    "swipe": "swipe up",
    "scroll_to_find": "scroll until you find <item>",
    "screenshot": "take a screenshot",
    "set_dnd": "set do not disturb to <mode>",
}

HOW = {
    "open_app": "list installed apps → Jev picks one → launch → confirm it is in front",
    "dumpsys": "list system services → Jev picks one → read it → Jev extracts the answer",
    "tap": "read the screen → Jev picks the element → safety gate → tap → verify",
    "long_press": "read the screen → Jev picks the element → safety gate → long-press → verify",
    "type_text": "read the screen → Jev picks the field and text → safety gate → type → verify",
    "toggle_service": "Jev picks the service and on/off → safety gate → run → verify live status",
    "keyevent": "Jev picks the exact key → safety gate → press",
    "swipe": "Jev picks the direction → safety gate → swipe",
    "set_dnd": "Jev picks the mode → safety gate → apply",
    "scroll_to_find": "scroll and re-read the screen until the item is visible",
    "screenshot": "capture the screen",
}

DETAIL_LABELS = {
    "package": "App", "service": "Service", "check_service": "Checked via", "element": "Element",
    "exit_code": "Exit code", "found": "Found", "attempts": "Attempts", "saved": "Saved to",
}


# --- small UI helpers ---------------------------------------------------------

def step(n: int, title: str) -> None:
    console.print()
    console.print(Rule(Text(f" Step {n} · {title} ", style=f"bold {ACCENT}"), style="dim", align="left"))


def ok(msg: str) -> None:
    console.print(f"  [green]✓[/] {msg}")


def warn(msg: str) -> None:
    console.print(f"  [yellow]![/] {msg}")


def fail(msg: str) -> None:
    console.print(f"  [red]✗[/] {msg}")


def row(label: str, content, *, bold: bool = False) -> None:
    """One label/value line; every goal's output shares these column widths, so wrapped text hangs aligned."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", width=16, style="bold" if bold else "dim")
    grid.add_column()
    grid.add_row(label, content)
    console.print(Padding(grid, (0, 0, 0, 2)))


def adb(*args: str, timeout: float = 20) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["adb", *args], capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", f"timed out after {timeout:.0f}s")


def banner() -> None:
    title = Text("jevdevice", style=f"bold {ACCENT}")
    title.append("  ·  plain-language control of a real Android phone", style="bold")
    body = Text.from_markup(
        "You type a goal like [bold]\"open the calculator\"[/]. jevdevice turns it into exactly "
        "[bold]one[/] real action on your phone, chosen only from what is actually there, "
        "safety-checked before it runs, and verified afterwards. If it isn't sure, it "
        "asks you, or declines to act."
    )
    console.print()
    console.print(Panel(body, title=title, title_align="left", border_style=ACCENT, padding=(1, 2)))


# --- setup --------------------------------------------------------------------

def check_prerequisites() -> None:
    step(1, "Checking this computer")
    if shutil.which("adb") is None:
        fail("adb (Android Debug Bridge) was not found.")
        console.print("    Install it with [bold]sudo apt install adb[/] (Linux) or "
                      "[bold]brew install android-platform-tools[/] (macOS), then rerun.")
        raise SystemExit(1)
    ok(adb("version").stdout.splitlines()[0])

    from jevdevice.common import load_env_file
    env_file = load_env_file()
    # Shadow checkpoint load takes minutes on CPU; an explicit JEV_SHADOW=1 still opts in.
    os.environ.setdefault("JEV_SHADOW", "0")
    engine = (os.environ.get("JEV_ENGINE") or "jev").strip().lower()
    if engine == "laya":
        ok("Judge engine: laya (runs locally, no API key needed)")
        return
    if os.environ.get("TYPESAFE_AI_API"):
        ok(f"TypeSafe API key found{f' in {env_file}' if env_file else ''}")
        return
    warn("No TypeSafe API key found (TYPESAFE_AI_API).")
    key = Prompt.ask("    Paste your key (input hidden, used for this session only)", password=True, console=console).strip()
    if not key:
        fail("A key is required for the hosted Jev engine.")
        raise SystemExit(1)
    os.environ["TYPESAFE_AI_API"] = key
    ok("API key set for this session")


def show_phone_instructions() -> None:
    step(2, "Prepare your phone")
    steps = Table.grid(padding=(0, 2))
    steps.add_column(style=f"bold {ACCENT}", justify="right")
    steps.add_column()
    steps.add_row("1", "Connect the phone to the [bold]same Wi-Fi network[/] as this computer.")
    steps.add_row("2", "Enable Developer options: [bold]Settings → About phone[/] → tap [bold]Build number[/] 7 times.")
    steps.add_row("3", "Open [bold]Settings → System → Developer options[/] and turn on [bold]Wireless debugging[/].")
    steps.add_row("4", "Tap [bold]Wireless debugging[/] and note the [bold]IP address & Port[/] shown at the top.")
    console.print(Padding(steps, (0, 0, 0, 2)))
    console.print("  [dim]Menu names vary slightly between phone makers; search Settings for \"Wireless debugging\".[/]")
    console.print()
    Prompt.ask("  Press [bold]Enter[/] when Wireless debugging is on", default="", show_default=False, console=console)


def _default(value) -> dict:
    """rich returns a `default=None` verbatim on empty input; omit it so the prompt re-asks instead."""
    return {} if value is None else {"default": value}


def ask_ip(default: str | None) -> str:
    while True:
        value = Prompt.ask("  Phone IP address", **_default(default), console=console).strip()
        try:
            return str(ipaddress.ip_address(value))
        except ValueError:
            fail(f"\"{value}\" is not a valid IP address (it looks like 192.168.1.23).")


def ask_port(label: str, default: int | None) -> int:
    while True:
        port = IntPrompt.ask(f"  {label}", **_default(default), console=console)
        if 1 <= port <= 65535:
            return port
        fail("A port is a number between 1 and 65535.")


def pair(ip: str) -> None:
    console.print()
    console.print("  [bold]First-time pairing[/]: on the phone, inside Wireless debugging, tap")
    console.print("  [bold]Pair device with pairing code[/]. A [bold]6-digit code[/] and a [bold]pairing port[/] appear.")
    console.print("  [dim](The pairing port is different from the connection port.)[/]")
    pair_port = ask_port("Pairing port", None)
    code = Prompt.ask("  6-digit pairing code", console=console).strip()
    with console.status(f"  Pairing with {ip}:{pair_port}…", spinner="dots"):
        result = adb("pair", f"{ip}:{pair_port}", code)
    message = (result.stdout + result.stderr).strip()
    if "Successfully paired" in message:
        ok("Paired. This computer is now trusted by the phone.")
    else:
        fail(f"Pairing failed: {message or 'no response from adb'}")


def connect_phone() -> str:
    step(3, "Connect")
    default_ip, default_port = None, None
    saved = os.environ.get("ANDROID_SERIAL", "")
    if ":" in saved:
        host, _, port = saved.rpartition(":")
        default_ip, default_port = host, int(port) if port.isdigit() else None
    ip = ask_ip(default_ip)
    port = ask_port("Port", default_port)
    while True:
        serial = f"{ip}:{port}"
        with console.status(f"  Connecting to {serial}…", spinner="dots"):
            reply = adb("connect", serial)
            state = adb("-s", serial, "get-state").stdout.strip()
        if state == "device":
            ok(f"Connected to {serial}")
            return serial
        if state == "unauthorized":
            warn("The phone is asking for permission. Tap [bold]Allow[/] on the phone's screen.")
        else:
            fail(f"Could not connect: {(reply.stdout + reply.stderr).strip() or 'no response'}")
        console.print("    [bold]r[/] retry  [bold]p[/] pair (first time)  [bold]e[/] edit IP/port  [bold]q[/] quit")
        choice = Prompt.ask("  Choose", choices=["r", "p", "e", "q"], default="r", console=console)
        if choice == "q":
            raise SystemExit(0)
        if choice == "p":
            pair(ip)
        elif choice == "e":
            ip, port = ask_ip(ip), ask_port("Port", port)


def describe_phone(serial: str) -> str:
    maker, model, release = (adb("-s", serial, "shell", "getprop", p).stdout.strip()
                             for p in ("ro.product.manufacturer", "ro.product.model", "ro.build.version.release"))
    return f"{maker.title()} {model} · Android {release}"


# --- running one goal ---------------------------------------------------------

class RecordingDevice:
    """The real device, plus a log of every shell command sent to it, so the demo
    can show exactly what ran instead of summarizing it."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.commands: list[str] = []
        self.screen_reads = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def dump_hierarchy(self) -> str:
        self.screen_reads += 1
        return await self._inner.dump_hierarchy()

    async def run(self, command: str, *args, **kwargs):
        if await self._may_capture(command):
            await camera_countdown()
        self.commands.append(command)
        return await self._inner.run(command, *args, **kwargs)

    async def _may_capture(self, command: str) -> bool:
        """A camera key, or any tap/key press while a camera app is in front, can take a photo."""
        if "KEYCODE_CAMERA" in command:
            return True
        if not command.startswith(("input tap", "input keyevent")):
            return False
        focus = await self._inner.run("dumpsys window | grep -m1 mCurrentFocus")
        return "camera" in focus.stdout.lower()

    async def run_binary(self, command: str, *args, **kwargs) -> bytes:
        self.commands.append(command)
        return await self._inner.run_binary(command, *args, **kwargs)

    async def window_size(self):
        return await self._inner.window_size()


async def camera_countdown(seconds: int = 3) -> None:
    console.print(Padding(Panel("[bold]This action may take a photo with the phone's camera.[/]",
                                border_style="red", padding=(0, 1)), (0, 0, 0, 2)))
    for remaining in range(seconds, 0, -1):
        console.print(f"  [bold red]Camera in {remaining}…[/]")
        await asyncio.sleep(1)


def latest_decision(goal: str, *, kind_pick: bool) -> dict:
    """Answers of the newest decision for this goal: the kind pick's, or else the last one
    after it. Journal rows are written synchronously, so the decision is already on disk."""
    from jevdevice.journal.decision_log import journal_dir
    files = sorted(Path(journal_dir()).glob("journal-*.jsonl"))
    for line in reversed(files[-1].read_text(encoding="utf-8").splitlines() if files else []):
        if goal not in line:
            continue
        decision = json.loads(line)
        scope_ok = (decision.get("scope") or {}).get("goal") == goal and not decision["scope"].get("shadow_of")
        if decision.get("type") == "decision" and scope_ok and (decision.get("phase") == "kind") == kind_pick:
            return decision.get("answers") or {}
    return {}


def explain_refusal(goal: str, *, kind_pick: bool, floor: float) -> list[tuple[str, str]]:
    """Rows saying what Jev weighed before declining: the options it chose between (or the
    single best match) and, when one was asked, its does-anything-fit check."""
    answers = latest_decision(goal, kind_pick=kind_pick)
    rows: list[tuple[str, str]] = []
    for answer in answers.values():
        options = sorted(((k, p) for k, p in (answer.get("probabilities") or {}).items() if p >= 0.05),
                         key=lambda kp: -kp[1])
        if len(options) > 1:
            rows.append(("Options", "[dim]Jev was split between:[/]"))
            rows += [("", name(o, p)) for o, p in options[:4]]
        elif options:
            rows.append(("Best match", f"{name(options[0][0])}"))
        if options:
            break
    if (fit := (answers.get("any_fit") or {}).get("noul")) is not None and fit < floor:
        what = "any action type" if kind_pick else "any option"
        rows.append(("Fit check", f"{fit:.0%} sure {what} fits this goal [dim](needs {floor:.0%})[/]"))
    return rows


def name(candidate: str, p: float | None = None) -> str:
    """Human name for a candidate: kinds and packages as is, screen elements by label."""
    from jevdevice.execution.dispatch import ACTION_KINDS
    pct = f" {p:.0%}" if p is not None else ""
    if candidate in ACTION_KINDS:
        return f"[{ACCENT}]{candidate}[/]{pct} [dim]· {ACTION_KINDS[candidate]}[/]"
    return f"{readable(candidate)}[dim]{pct}[/]"


_ELEMENT = re.compile(r"(?:(?:text|content-desc|resource-id|class)='[^']*'\s*)+")


def readable(value: str) -> str:
    """Rewrite raw element attribute blobs as `"visible label" (resource id)`."""
    def label(match: re.Match) -> str:
        blob = match.group(0)
        shown = next((m.group(1) for attr in ("text", "content-desc")
                      if (m := re.search(rf"{attr}='([^']+)'", blob))), None)
        rid = re.search(r"resource-id='[^']*?(\w+)'", blob)
        if shown and len(shown) > 48:
            shown = shown[:47] + "…"
        parts = [f"\"{shown}\"" if shown else "unlabelled", f"({rid.group(1)})" if rid else ""]
        return " ".join(p for p in parts if p) + " "
    return _ELEMENT.sub(label, value).strip().replace('""', '"')


KEY_NAMES = {"66": "enter", "67": "delete", "123": "end of line"}


def compact_command(command: str, limit: int | None = None) -> str:
    """Collapse runs of a repeated argument (`67 67 67 …` → `67 ×250`) and name the key codes."""
    def collapse(match: re.Match) -> str:
        return f"{match.group(1)} ×{len(match.group(0).split())}"
    short = re.sub(r"\b(\S+)(?: \1\b){3,}", collapse, command)
    codes = {c for part in short.split("&&") if part.strip().startswith("input keyevent")
             for c in re.findall(r"\b\d+\b", part)}
    legend = ", ".join(f"{c} = {KEY_NAMES[c]}" for c in sorted(codes, key=int) if c in KEY_NAMES)
    if limit and len(short) > limit:
        short = short[:limit - 1] + "…"
    return f"{short}  ({legend})" if legend else short


def humanize(reason: str) -> str:
    """Pipeline reasons carry raw thresholds; say them as percentages."""
    reason = re.sub(r"confidence (\d\.\d+) below (\d\.\d+)",
                    lambda m: f"Jev was {float(m.group(1)):.0%} sure; it needs {float(m.group(2)):.0%} to act", reason)
    return re.sub(r"\(best (\d\.\d+) below (\d\.\d+)\)",
                  lambda m: f"(best {float(m.group(1)):.0%}, needs {float(m.group(2)):.0%})", reason)


def jev_requests(jev) -> int:
    usage = getattr(jev, "usage", None)
    return usage.snapshot().requests if usage is not None else 0


def ask_approval(pending) -> bool:
    body = Table.grid(padding=(0, 2))
    body.add_column(style="dim", justify="right")
    body.add_column()
    action = readable(pending.chosen_label)
    body.add_row("Action", action)
    body.add_row("Command", Text(compact_command(pending.command.command), style="bold"))
    why = readable(pending.command.rationale or "")
    if why and re.sub(r'[\s"]', "", action) not in re.sub(r'[\s"]', "", why):
        body.add_row("Why", why)
    if pending.gate_result is not None and pending.gate_result.confidence is not None:
        body.add_row("Safety check", f"{pending.gate_result.confidence:.0%} sure this does only what was asked "
                                     "(below the auto-run bar)")
    console.print(Padding(Panel(body, title="[bold yellow]Your approval is needed[/]", title_align="left",
                              border_style="yellow", padding=(0, 1)), (0, 0, 0, 2)))
    return Confirm.ask("  Run this command on the phone?", default=False, console=console)


async def run_goal(jev, device: RecordingDevice, goal: str) -> None:
    from jevdevice.actions.services import take_screenshot
    from jevdevice.budget import current_profile
    from jevdevice.execution.dispatch import (
        ACTION_KINDS,
        KIND_TABLE,
        pick_kind,
        response_for,
        run_kind,
    )
    from jevdevice.journal import outcomes
    from jevdevice.journal.decision_log import goal_scope

    console.print()
    console.print(Text.assemble(("▶ ", f"bold {ACCENT}"), (f"\"{goal}\"", "bold")))
    device.commands.clear()
    device.screen_reads = 0
    calls_before, t0 = jev_requests(jev), time.monotonic()
    quiet = io.StringIO()  # swallows the pipeline's own debug prints; results are rendered below instead

    with goal_scope(goal):
        with console.status("  Understanding the goal (reading the screen, choosing an action type)…", spinner="dots"), \
                contextlib.redirect_stdout(quiet):
            pick = await pick_kind(jev, goal, device, verbose=False)
        floor = current_profile(jev.name).noul_floor
        if pick.kind is None:
            explained = explain_refusal(goal, kind_pick=True, floor=floor)
            row("Understand", "[yellow]no action type is a confident fit[/]", bold=True)
            for label, content in explained:
                row(label, content)
            row("Result", f"[yellow]✗ Did not act[/] [dim]· {humanize('; '.join(pick.reasons))}; nothing was changed[/]",
                bold=True)
            options = [k for k, p in sorted((latest_decision(goal, kind_pick=True).get("kind") or {})
                                            .get("probabilities", {}).items(), key=lambda kp: -kp[1])[:2]]
            hints = [PHRASING[k] for k in options if k in PHRASING]
            if hints:
                row("Tip", "say which one you mean, e.g. " + " or ".join(f"[bold]\"{h}\"[/]" for h in hints))
            return _footer(jev, device, calls_before, t0)
        row("Understand", f"[{ACCENT}]{pick.kind}[/] [dim]· {ACTION_KINDS[pick.kind]} "
                          f"({pick.confidence:.0%} confident)[/]", bold=True)
        row("Act", f"[dim]{HOW.get(pick.kind, '')}[/]", bold=True)

        if pick.kind == "screenshot":
            with console.status("  Capturing the screen…", spinner="dots"):
                png = await take_screenshot(device)
            SCREENSHOT_DIR.mkdir(exist_ok=True)
            path = SCREENSHOT_DIR / f"screenshot-{time.strftime('%Y%m%d-%H%M%S')}.png"
            path.write_bytes(png)
            return _render({"status": "ok", "saved": str(path)}, jev, device, calls_before, t0)

        status = console.status("  Working on the phone…", spinner="dots")

        async def on_pending(goal, kind, resume_arg, confidence, pending, verify, *, label=None):
            """The MCP server's device_approve resume path, asked in the terminal instead."""
            status.stop()
            approved = ask_approval(pending)
            call_id = pending.gate_result.call_id if pending.gate_result else None
            if not approved:
                outcomes.record_action(device=device, call_id=call_id, key=kind, gate=pending.gate_result,
                                       status="not_executed", decision="deny")
                return {"status": "not_executed"}
            status.start()
            proposal = SimpleNamespace(element=resume_arg, service=resume_arg, confidence=confidence)
            outcome, edge = await outcomes.graph_edge_around(device, lambda: KIND_TABLE[kind].execute(
                jev, device, goal, proposal, pending.command, verify=verify, verbose=False))
            response = response_for(kind, outcome, jev.name)
            outcomes.record_action(device=device, call_id=call_id, key=kind, gate=pending.gate_result,
                                   executed_command=pending.command.command, response=response,
                                   graph_edge=edge, decision="approve", label=label)
            return response

        status.start()
        try:
            with contextlib.redirect_stdout(quiet):
                response = await run_kind(jev, device, pick.kind, goal, verify=True, on_pending=on_pending)
        finally:
            status.stop()
    explained = explain_refusal(goal, kind_pick=False, floor=floor) if response.get("status") == "escalated" else []
    _render(response, jev, device, calls_before, t0, kind=pick.kind, explained=explained)


async def run_goal_safely(jev, device: RecordingDevice, goal: str) -> None:
    try:
        await run_goal(jev, device, goal)
    except Exception as exc:  # noqa: BLE001 -- one failed goal must not end the session
        fail(f"Something went wrong: {type(exc).__name__}: {exc}")


def _render(response: dict, jev, device: RecordingDevice, calls_before: int, t0: float, *,
            kind: str | None = None, explained: list[tuple[str, str]] = ()) -> None:
    verified = "[green]✓ Done[/] [dim]· verified on the device[/]"
    if kind in EXIT_CODE_ONLY:
        verified = "[green]✓ Sent[/] [dim]· the phone accepted it (exit code only; the effect isn't re-checked)[/]"
    headline = {
        "ok": verified,
        "unverified": "[yellow]! Ran, but the result could not be confirmed[/]",
        "escalated": "[yellow]✗ Did not act[/] [dim]· not confident enough; nothing was changed[/]",
        "not_executed": "[dim]✗ Skipped · you declined; nothing was changed[/]",
    }.get(response.get("status"), f"[yellow]{response.get('status')}[/]")
    row("Result", headline, bold=True)
    for key, value in (response.get("answer") or {}).items():
        row("Answer", Text.assemble((f"{key.replace('_', ' ')}: ", "dim"), (str(value), "bold")))
    for key, label in DETAIL_LABELS.items():
        if response.get(key) not in (None, "", []):
            row(label, readable(str(response[key])))
    if isinstance(response.get("satisfied"), float):
        row("Goal met (Jev)", f"{response['satisfied']:.0%}")
    for reason in dict.fromkeys(response.get("reasons") or []):
        row("Why not", humanize(reason))
    for label, content in explained:
        row(label, content)
    if any(label == "Options" for label, _ in explained):
        row("Tip", "be more specific: name the one you mean, by its on-screen label")
    _footer(jev, device, calls_before, t0)


def _footer(jev, device: RecordingDevice, calls_before: int, t0: float) -> None:
    counts = Counter(device.commands)
    shown = list(counts)
    for i, command in enumerate(shown[:4]):
        times = f" ×{counts[command]}" if counts[command] > 1 else ""
        row("Sent to phone" if i == 0 else "", f"[dim]$ {compact_command(command, limit=100)}{times}[/]")
    if len(shown) > 4:
        row("", f"[dim]… and {len(shown) - 4} more[/]")
    calls, reads = jev_requests(jev) - calls_before, device.screen_reads
    row("", f"[dim]{calls} Jev call{'s' * (calls != 1)} · {reads} screen read{'s' * (reads != 1)} · "
            f"{time.monotonic() - t0:.1f}s[/]")


KNOB_NAMES = {
    "gate_threshold": "Safety check", "min_fit": "Candidate fit",
    "min_confidence": "Pick confidence", "min_margin": "Pick margin",
}


def recalibrate(jev) -> None:
    """Refit the thresholds from this and earlier sessions' labelled outcomes: stricter-only
    (loosening needs recorded human sign-off), zero model calls."""
    from typesymbolic.calibrate import DEFAULT_MIN_LABELS

    from jevdevice.calibrate import continuous

    with console.status("  Recalibrating from the journal (no Jev calls)…", spinner="dots"):
        results = continuous.run(engine=jev.name, human_signoff=False, quiet=True)
    console.print()
    console.print(Rule(Text(" Calibration ", style=f"bold {ACCENT}"), style="dim", align="left"))
    for knob, r in results.items():
        if knob == "min_margin":
            state = "[dim]fixed · margins aren't labelled (one scale per answer)[/]"
        elif r.applied:
            state = f"[green]tightened[/] {r.before:.2f} → [bold]{r.after:.2f}[/]"
        elif r.n_labels < DEFAULT_MIN_LABELS:
            state = f"[dim]{r.after:.2f} · collecting labels ({r.n_labels}/{DEFAULT_MIN_LABELS})[/]"
        else:
            state = f"{r.after:.2f} [dim]· unchanged ({r.n_labels} labels)[/]"
        row(KNOB_NAMES.get(knob, knob), state)
    console.print("  [dim]Each verified or failed action adds labels; thresholds only ever get stricter here.[/]")


# --- modes --------------------------------------------------------------------

async def guided_tour(jev, device) -> None:
    console.print()
    console.print(Rule(Text(" Guided tour ", style=f"bold {ACCENT}"), style="dim", align="left"))
    console.print("  [dim]Keep your phone unlocked and in view. Each command waits for you.[/]")
    for i, (goal, about) in enumerate(TOUR, 1):
        console.print()
        console.print(f"  [bold {ACCENT}]{i}/{len(TOUR)}[/]  [bold]\"{goal}\"[/]  [dim]{about}[/]")
        choice = Prompt.ask("  [bold]Enter[/] run · [bold]s[/] skip · [bold]q[/] back to menu",
                            choices=["", "s", "q"], default="", show_choices=False, show_default=False, console=console)
        if choice == "q":
            return
        if choice == "":
            await run_goal_safely(jev, device, goal)
    console.print()
    ok("Tour complete. Try free mode next to type your own commands.")


async def free_mode(jev, device) -> None:
    console.print()
    console.print(Rule(Text(" Free mode ", style=f"bold {ACCENT}"), style="dim", align="left"))
    console.print("  Type what you want the phone to do, in plain language. One action per command.")
    console.print("  [dim]Examples: " + " · ".join(f"\"{e}\"" for e in EXAMPLES[:3]) + "[/]")
    console.print("  [dim]Type [bold]help[/bold] for more examples, or press Enter on an empty line to go back.[/]")
    while True:
        console.print()
        goal = Prompt.ask(f"  [bold {ACCENT}]›[/]", default="", show_default=False, console=console).strip()
        if goal.lower() in ("", "back", "menu", "exit", "quit", "q"):
            return
        if goal.lower() == "help":
            for example in EXAMPLES:
                console.print(f"    [dim]•[/] {example}")
            console.print("  [dim]Multi-step tasks go one step at a time: \"open settings\", then \"tap Wi-Fi\".[/]")
            continue
        await run_goal_safely(jev, device, goal)


async def menu(jev, device) -> None:
    while True:
        console.print()
        console.print(Rule(Text(" What would you like to do? ", style="bold"), style="dim", align="left"))
        console.print(f"  [bold {ACCENT}]1[/]  Guided tour   [dim]run a few ready-made commands, one at a time[/]")
        console.print(f"  [bold {ACCENT}]2[/]  Free mode     [dim]type your own commands[/]")
        console.print(f"  [bold {ACCENT}]c[/]  Calibration   [dim]refit thresholds from what has been verified so far[/]")
        console.print(f"  [bold {ACCENT}]q[/]  Quit")
        choice = Prompt.ask("  Choose", choices=["1", "2", "c", "q"], default="1", console=console)
        if choice == "q":
            return
        if choice == "c":
            recalibrate(jev)
            continue
        await (guided_tour if choice == "1" else free_mode)(jev, device)


async def main() -> None:
    banner()
    check_prerequisites()
    show_phone_instructions()
    serial = connect_phone()

    step(4, "Start jevdevice")
    with console.status("  Loading the engine…", spinner="dots"):
        from jevdevice.common import bootstrap
        jev, raw_device = bootstrap(serial)
    ok(f"Phone: {describe_phone(serial)}")
    ok(f"Judge engine: {jev.name}")
    device = RecordingDevice(raw_device)
    try:
        with console.status("  Starting the on-phone screen reader (first run can take ~30s)…", spinner="dots"):
            await raw_device.dump_hierarchy()
        ok("Screen reader ready")
    except Exception as exc:  # noqa: BLE001 -- screen-based kinds then fail closed; the rest still work
        warn(f"Screen reader did not start ({type(exc).__name__}); screen-based actions may not work.")

    try:
        await menu(jev, device)
        recalibrate(jev)
    finally:
        await jev.aclose()
    console.print()
    console.print(f"  [dim]Every decision and action was journaled to "
                  f"{os.environ.get('JEV_JOURNAL_DIR', '~/.jevdevice/tsjournal')}.[/]")
    console.print("  [bold]Thanks for trying jevdevice.[/]\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        console.print("\n  [dim]Stopped. Goodbye.[/]\n")
