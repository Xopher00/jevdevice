"""Voice in, jevdevice out: record a spoken goal, transcribe locally with
whisper.cpp, then hand it to dispatch.py's tool dispatch — a spoken goal
isn't necessarily an app launch. The only content unique to this script is
the audio capture.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile

from jevdevice.common import bootstrap
from jevdevice.dispatch import run_toolkit

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
    jev, transport = bootstrap()

    goal = record_and_transcribe()
    print(f"\ntranscribed goal: {goal!r}\n")
    if not goal:
        print("nothing transcribed, exiting")
        return

    await run_toolkit(jev, transport, goal)


if __name__ == "__main__":
    asyncio.run(main())
