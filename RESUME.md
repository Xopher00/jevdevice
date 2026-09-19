# RESUME

Read the last entries of LOGBOOK.md first.

## Where things stand (2026-09-19)

The primary interface is now the MCP server, not a CLI script: `jevdevice-mcp` (entry point in
`pyproject.toml`) exposes three tools -- `device_do(goal, ...)` (Jev picks one atomic action kind
via `dispatch.pick_kind`/`ACTION_KINDS` and runs exactly that one action, gated, with a proof
screenshot attached for visually-meaningful kinds), `device_screenshot()` (on demand, ungated),
and `device_approve(thread_id, decision)` (resolves a `needs_approval` verdict). Code changes to
any `.py` file need a client-side MCP reconnect before they take effect live -- there is no
programmatic way to trigger that reconnect from inside a session.

Module layout in `src/jevdevice/`:
- `transport.py`: `AdbTransport` -- shells real adb commands, and lazily holds a persistent
  `uiautomator2` connection for fast hierarchy dumps/screen size (`dump_hierarchy`/`window_size`).
- `jev.py`, `matching.py`, `narrowing.py`: the Jev client and the real-enumeration -> two-round
  semantic narrow -> gate machinery every propose_* function is built on.
- `gate.py`: deny-list/read-only classification (argv-shape based, not substring matching) +
  the Jev safety Noul every mutating command passes before running.
- `common.py`: `bootstrap()` (env/API-key + `AdbTransport`) and `gated` (the confidence-gate
  wrapper around one Jev Choice pick) -- small utilities nearly every other module imports.
- `elements.py`: parses a real `uiautomator2` XML dump into on-screen elements
  (`parse_actionable_elements`/`parse_editable_elements`/`parse_long_clickable_elements`/
  `parse_all_elements`), plus `dump_screen`/`foreground_package`.
- `ui.py`: propose/execute for tap/long_press/type/swipe/scroll_to_find over elements.py's real
  elements, plus the bounded `_ELEMENT_CACHE`.
- `app_launch.py`: `launch_app_for_goal` -- real package listing -> narrow -> gate -> `monkey`
  launch -> Jev-verified retry.
- `services.py`: propose/execute for system-service actions -- toggle a radio, press a key, set
  Do Not Disturb, read a dumpsys service.
- `dispatch.py`: `ACTION_KINDS`/`pick_kind` (which one atomic kind a goal wants) and `KIND_TABLE`
  (normalizes each kind's propose/execute so `run_toolkit` (CLI) and `mcp_server.py`'s
  `device_do`/`device_approve` (MCP) share one dispatch instead of independently drifting ones).
  `uv run python -m jevdevice.dispatch "<goal>"` runs one goal from the CLI, blocking-prompt style.
- `mcp_server.py`: the MCP server itself, built on the above.
- `voice_prove_it.py`: records a spoken goal and calls `dispatch.run_toolkit` with it.
- `calibrate/`: standalone diagnostics (`gate.py`, `gate_taps.py`, `narrowing.py`) for tuning Jev
  wording/thresholds against hand-picked live cases -- not imported by anything, run directly.

`prove_it.py` (the original single-purpose launch proof) is retired to `experiment/`.

Known wrinkle: none currently open.
