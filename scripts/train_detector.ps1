# scripts/train_detector.ps1
# Train the YOLOv8 dental detector on Windows.
#
# Usage (from repo root in PowerShell):
#   .\scripts\train_detector.ps1
#   .\scripts\train_detector.ps1 -Config configs\detector.yaml
#
param(
    [string]$Config = "configs\detector.yaml"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Push-Location $RepoRoot
try {
    Write-Host "=== Dental Model: Detector Training ===" -ForegroundColor Cyan
    Write-Host "Repo root : $RepoRoot"
    Write-Host "Config    : $Config"
    Write-Host ""

    Write-Host ">>> Syncing environment..." -ForegroundColor Yellow
    & "$env:USERPROFILE\.local\bin\uv.exe" sync --all-extras --dev

    Write-Host ""
    Write-Host ">>> Checking GPU..." -ForegroundColor Yellow
    & "$env:USERPROFILE\.local\bin\uv.exe" run python -c @"
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_memory / 1e9:.2f} GB')
"@

    Write-Host ""
    Write-Host ">>> Starting YOLOv8 training..." -ForegroundColor Yellow
    & "$env:USERPROFILE\.local\bin\uv.exe" run python -m dental_model.detector.train --config $Config

    Write-Host ""
    Write-Host "=== Training complete. Weights saved under models\detector_runs\ ===" -ForegroundColor Green
} finally {
    Pop-Location
}
