#!/usr/bin/env bash
# ollama 0.32.15 checks <cwd>/build/lib/ollama/llama-server; copy the working
# user install there since system ollama ships without one.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build/lib/ollama
cp -r ~/.local/ollama/lib/ollama/. build/lib/ollama/
chmod +x build/lib/ollama/llama-server
echo "Run: OLLAMA_HOST=127.0.0.1:11500 ollama serve   (from this repo's root)"
