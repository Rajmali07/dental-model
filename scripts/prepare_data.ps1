Write-Host "=== Phase 0: Data Extraction & Label Unification ==="
uv run python -m dental_model.data.extract
uv run python -m dental_model.data.unify_labels
Write-Host "=== Phase 0 Complete! ==="
