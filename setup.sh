#!/usr/bin/env bash
# BitWiser — environment bootstrap for Apple Silicon.
# Builds llama.cpp with Metal acceleration and installs Python deps.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENDOR="$ROOT/vendor"
LLAMA="$VENDOR/llama.cpp"

mkdir -p "$VENDOR"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required. Install with: brew install cmake"
  exit 1
fi

if [ ! -d "$LLAMA" ]; then
  echo "==> Cloning llama.cpp"
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA"
else
  echo "==> Updating llama.cpp"
  git -C "$LLAMA" pull --ff-only
fi

echo "==> Configuring with Metal"
cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_METAL=ON -DLLAMA_CURL=OFF

echo "==> Building (Release)"
cmake --build "$LLAMA/build" --config Release -j

echo "==> Installing Python deps"
python3 -m pip install --upgrade pip
python3 -m pip install -r "$ROOT/requirements.txt"

echo
echo "Done. Binaries:"
ls "$LLAMA/build/bin" | grep -E '^(llama-quantize|llama-cli)$' || true
