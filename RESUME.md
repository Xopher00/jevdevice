# RESUME

Read PLAN.md first, then the last entries of LOGBOOK.md.

## Where things stand (2026-09-19)

M0 is done. Next is M1: the engine on the PC, questions only (PLAN.md section 7).
No engine code exists yet. `src/jevdevice/` holds `transport.py`, `jev.py`, `matching.py`
(all live-verified) and `gate.py`, `grounding.py` (to be folded into `judge.py` at M1).

## The rule

Nobody hand-writes device commands, anywhere. PLAN.md section 9 is the complete list of what is
hand-written. Section 10 has the guard tests to write first at M1. Do not run device commands by
hand to check things. Do not reword a prompt because one case failed.

## Environment

- Python: `cd jevdevice && uv sync --all-groups`
- Jev: OpenRouter key is `OPENROUTER_API_KEY` in `../.env`. Endpoint
  `https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13` (client: `src/jevdevice/jev.py`).
  Load it without printing it: `export OPENROUTER_API_KEY=$(grep '^OPENROUTER_API_KEY=' ../.env | cut -d= -f2-)`
- Small model: ollama, user-owned daemon on port 11500. The system ollama lacks `llama-server`;
  `scripts/setup_ollama.sh` copies the working libs into `build/` (gitignored), then from the repo
  root: `OLLAMA_HOST=127.0.0.1:11500 ollama serve`. Pulled: `qwen2.5-coder:3b`. A root-owned
  ollama (pid seen: 4517) also exists; leave it alone.
- Phone: Samsung S24, wireless adb, `adb connect 192.168.100.11:46585`. The port changes whenever
  wifi reconnects. Never test wifi or mobile-data actions over this link.
- Sandbox: `bwrap` is installed.
- Voice: `~/tools/whisper.cpp/build/bin/whisper-cli`, model `ggml-small.bin`. Always play the
  audible cue before recording.

## Open items for M1

- Guard tests first (PLAN.md section 10).
- Decide how never-seen PC commands get sandboxed (section 6 rule 6) without hand-writing device
  knowledge into `src/`.
- M2 needs a spending estimate approved before the baseline run.
