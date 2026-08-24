#!/usr/bin/env bash
set -e
echo "=== Phase 0: Data Extraction & Label Unification ==="
uv run python -m dental_model.data.extract
uv run python -m dental_model.data.unify_labels
echo "=== Phase 0 Complete! ==="
