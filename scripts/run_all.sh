#!/usr/bin/env bash
# Run the whole pipeline end to end.
#
#   ./scripts/run_all.sh                      # default (CPU pilot) config
#   ./scripts/run_all.sh configs/smoke.yaml   # fast sanity run
#
# Safe to re-run: training resumes, and every later stage only reads saved
# artefacts, so an interrupted run picks up where it stopped.

set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
PY="${PY:-python}"
cd "$(dirname "$0")/.."

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

step "0/7  dataset";       $PY scripts/00_build_dataset.py       --config "$CONFIG"
step "1/7  baselines";     $PY scripts/01_train_baselines.py     --config "$CONFIG"
step "2/7  transformers";  $PY scripts/02_train_transformers.py  --config "$CONFIG" --resume
step "3/7  evaluation";    $PY scripts/03_evaluate.py            --config "$CONFIG"
step "4/7  faithfulness";  $PY scripts/04_faithfulness.py        --config "$CONFIG"
step "5/7  stability";     $PY scripts/05_stability.py           --config "$CONFIG"
step "6/7  plausibility";  $PY scripts/06_plausibility.py        --config "$CONFIG"
step "7/7  figures";       $PY scripts/07_figures_tables.py      --config "$CONFIG"

printf '\n\033[1mDone.\033[0m  metrics: results/metrics  figures: results/figures  tables: results/tables\n'
printf 'Serve it:  uvicorn backend.app.main:app --port 8000   and   cd frontend && npm run dev\n'
