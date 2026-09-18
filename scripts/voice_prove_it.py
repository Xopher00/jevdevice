"""Voice in, jevdevice out: record a spoken goal, transcribe locally with
whisper.cpp, then run the exact mechanism proven in prove_it.py unchanged.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prove_it import (
    confidence_gate,
    fuzzy_narrow,
    parse_package_list,
    verify_with_retry,
)

from jevdevice.jev import Choice, JevClient
from jevdevice.transport import AdbTransport

SERIAL = "192.168.100.11:46585"
WHISPER_BIN = os.path.expanduser("~/tools/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.path.expanduser("~/tools/whisper.cpp/models/ggml-small.bin")
RECORD_SECONDS = 6


def _beep(freq: int, duration: float = 0.25) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        beep_path = f.name
    subprocess.run(["sox", "-n", "-r", "16000", "-c", "1", beep_path, "synth", str(duration), "sine", str(freq)], check=True, capture_output=True)
    subprocess.run(["paplay", beep_path], check=True, capture_output=True)
    os.unlink(beep_path)


def record_and_transcribe() -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    print(f"Recording will start on the beep and last {RECORD_SECONDS}s; a second beep means stop.")
    _beep(880)  # audio cue: print() output isn't shown live, so this is the real signal
    subprocess.run(
        ["arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(RECORD_SECONDS), wav_path],
        capture_output=True, check=True,
    )
    _beep(440)
    result = subprocess.run(
        [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav_path, "-nt", "-np"],
        capture_output=True, text=True, check=True,
    )
    os.unlink(wav_path)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip() and not line.startswith(("main:", "whisper_", "system_info"))]
    return " ".join(lines).strip()


async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    goal = record_and_transcribe()
    print(f"\ntranscribed goal: {goal!r}\n")
    if not goal:
        print("nothing transcribed, exiting")
        return

    transport = AdbTransport(SERIAL)
    jev = JevClient(api_key)

    probe = await transport.run("pm list packages")
    packages = parse_package_list(probe.stdout)
    narrowed = fuzzy_narrow(goal, packages, limit=20)
    print(f"narrowed to {len(narrowed)} candidates from {len(packages)} real packages")

    criteria = {p: None for p in narrowed}
    answers = await jev.ask(
        {"goal": goal, "candidate_packages": narrowed},
        {"pick": Choice(instructions="Which package best satisfies the spoken goal?", criteria=criteria)},
    )
    pick = answers["pick"]
    print(f"Jev picked: {pick.choice} (confidence {pick.confidence:.2f})")

    ok, reason = confidence_gate(pick.probabilities, pick.confidence)
    if not ok:
        print(f"=== ESCALATED === {reason}; not acting on an unsure pick")
        return

    await transport.run(f"monkey -p {pick.choice} 1")
    ok, satisfied, attempts, focus_line = await verify_with_retry(jev, transport, goal, pick.choice)
    print(f"focus line: {focus_line.strip()}")
    print(f"goal met: {ok} (noul={satisfied:.2f}, after {attempts} attempt(s))")


if __name__ == "__main__":
    asyncio.run(main())
