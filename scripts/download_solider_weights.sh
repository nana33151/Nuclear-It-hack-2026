#!/usr/bin/env bash
# Download SOLIDER Market ReID checkpoint (Swin-S, ~190MB) into weights/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/weights"
OUT="$ROOT/weights/swin_s_market.pth"
URL="https://huggingface.co/rotivrotiv/model-weights/resolve/main/solider-swin-s/swin_s_market.pth"
if [[ -f "$OUT" ]]; then
  echo "already exists: $OUT"
  exit 0
fi
echo "downloading $URL → $OUT"
curl -L --fail -o "$OUT" "$URL"
ls -lh "$OUT"
