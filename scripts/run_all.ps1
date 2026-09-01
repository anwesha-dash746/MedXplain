# Run the whole pipeline end to end (PowerShell).
#
#   .\scripts\run_all.ps1
#   .\scripts\run_all.ps1 -Config configs\smoke.yaml
#
# Safe to re-run: training resumes and later stages only read saved artefacts.

param([string]$Config = "configs/default.yaml", [string]$Py = "python")

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Step($label, $script) {
    Write-Host "`n=== $label ===" -ForegroundColor Cyan
    & $Py $script --config $Config @args
    if ($LASTEXITCODE -ne 0) { throw "$script failed with exit code $LASTEXITCODE" }
}

Step "0/7  dataset"      "scripts/00_build_dataset.py"
Step "1/7  baselines"    "scripts/01_train_baselines.py"

Write-Host "`n=== 2/7  transformers ===" -ForegroundColor Cyan
& $Py scripts/02_train_transformers.py --config $Config --resume
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Step "3/7  evaluation"   "scripts/03_evaluate.py"
Step "4/7  faithfulness" "scripts/04_faithfulness.py"
Step "5/7  stability"    "scripts/05_stability.py"
Step "6/7  plausibility" "scripts/06_plausibility.py"
Step "7/7  figures"      "scripts/07_figures_tables.py"

Write-Host "`nDone." -ForegroundColor Green
Write-Host "metrics: results/metrics   figures: results/figures   tables: results/tables"
Write-Host "Serve it:  uvicorn backend.app.main:app --port 8000   and   cd frontend; npm run dev"
