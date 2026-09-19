# jevdevice

jevdevice is an MCP server that lets an LLM agent control a real Android phone. It uses [Jev](https://docs.typesafe.ai), TypeSafe's System One decision model, to pick a real, runtime-discovered target for each action. The agent gives a goal in plain language, such as "tap the search button". Jev never sees a fixed list of buttons written in advance. It sees the real accessibility tree from the phone at that moment and picks from it.

## The core idea

Most phone-automation tools work from a hardcoded map. A script author writes down exact coordinates, resource IDs, or a fixed decision tree for each app. This breaks the moment an app updates its layout. It also does not generalize to an app the author never tested.

jevdevice does not hardcode any of that. Every action follows the same real pipeline:

1. Enumerate the real, current state (the on-screen accessibility tree, the installed app list, or a live `dumpsys` service).
2. Narrow that real list to a candidate with Jev, a small model built for exactly this kind of closed-set judgment.
3. Gate the resulting command with a second Jev check and a deny-list, before anything runs.
4. Run the command and, when asked, verify the result against the goal with another Jev check.

No step in this pipeline can select an option that does not exist on the real device at that moment. This makes the agent's action set correct by construction, not by a list someone kept up to date by hand.

## Architecture

The server exposes three tools over MCP:

- `device_do(goal, ...)`: runs exactly one real action for one goal. Jev first picks the kind of action the goal needs. The choices are: tap, type, swipe, long-press, scroll-to-find, press a key, open an app, toggle a radio, set Do Not Disturb, and read a system service. `device_do` then runs that one action and never chains a second one on its own. An LLM agent that wants a multi-step task still plans and sequences each `device_do` call itself.
- `device_screenshot()`: returns a real screenshot of the current screen.
- `device_approve(thread_id, decision)`: resolves a pending action that needed a decision.

Every command that changes device state passes a gate before it runs. The gate rejects a fixed list of dangerous command shapes outright, for example `rm`, `reboot`, or a factory wipe. It asks Jev a direct safety question about every other command. A command Jev is unsure about returns `needs_approval` with a `thread_id` instead of running. A caller then either approves it through `device_approve` or passes `auto_approve=True` up front. A rejected command never runs, no matter which mode the caller used.

The codebase splits along this pipeline:

| File | Role |
|---|---|
| `transport.py` | Runs real `adb` commands and holds a persistent `uiautomator2` connection for fast screen reads |
| `elements.py` | Parses the real accessibility tree into on-screen elements |
| `ui.py` | Picks a tap, long-press, type, swipe, or scroll-to-find target and gates the result |
| `services.py` | Picks and gates system-service actions: toggle a radio, press a key, set Do Not Disturb, read `dumpsys` |
| `app_launch.py` | Picks a real installed app for a goal and verifies it opened |
| `dispatch.py` | Picks which action kind a goal needs and dispatches to the matching propose/execute pair |
| `gate.py` | The deny-list and the Jev safety check every mutating command passes |
| `mcp_server.py` | The MCP server itself, built on the modules above |

## Why the read path is fast

A naive read of the screen shells out to `uiautomator dump`, which starts a fresh Android runtime process each time. On a real device this costs about 2.2 seconds per call, most of it process startup, not the actual read. jevdevice instead sideloads `uiautomator2`'s companion app once and keeps that connection open. A warm screen read then takes about 0.3 seconds.

## Setup

You need:

- Python 3.12 or later, and [uv](https://docs.astral.sh/uv/)
- `adb`, with an Android phone reachable over USB or wireless debugging
- A TypeSafe API key ([docs.typesafe.ai](https://docs.typesafe.ai))

Steps:

1. Clone this repository and run `uv sync`.
2. Connect your phone and confirm it shows up: `adb devices`.
3. Set your device's serial as an environment variable: `export ANDROID_SERIAL=<serial-from-adb-devices>`.
4. Set your TypeSafe API key: `export TYPESAFE_AI_API=<your-key>`.
5. Add `jevdevice-mcp` to your MCP client's server config, or run `uv run jevdevice-mcp` directly to check it starts.

The first real screen read after startup installs `uiautomator2`'s companion app on the phone automatically. This takes a few seconds once, and every read after that is fast.

## Running it without an LLM client

`uv run python -m jevdevice.dispatch "open the calculator"` runs one goal from the command line and prints the result. A command that needs approval prompts you directly in the terminal.

## Tests

`uv run pytest` runs the test suite. Most tests are pure unit tests with no device needed. `tests/test_mcp_approval_flow.py` is a live integration test and needs a real connected phone and a real API key.

## Safety

Jev's safety check and the deny-list run before every mutating command, with no exception. `auto_approve` only changes what happens after Jev returns `needs_approval` for a command it is unsure about. A command the deny-list rejects, or that Jev rejects outright, never runs, regardless of `auto_approve`.

## License

MIT. See `LICENSE`.
