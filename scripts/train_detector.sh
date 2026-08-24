#!/usr/bin/env bash
# scripts/train_detector.sh
# Train the YOLOv8 dental detector.
#
# Usage (from repo root):
#   bash scripts/train_detector.sh
#   bash scripts/train_detector.sh --config configs/detector.yaml
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"




CONFIG="${1:-configs/detector.yaml}"

echo "=== Dental Model: Detector Training ==="
echo "Repo root : $REPO_ROOT"
echo "Config    : $CONFIG"
echo ""

echo ">>> Syncing environment..."
uv sync --all-extras --dev

echo ""
echo ">>> Checking GPU..."
uv run python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_memory / 1e9:.2f} GB')
"

echo ""
echo ">>> Starting YOLOv8 training..."
uv run python -m dental_model.detector.train --config "$CONFIG"

echo ""
echo "=== Training complete. Weights saved under models/detector_runs/ ==="
