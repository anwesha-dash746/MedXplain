# MedXplain - one entry point for the whole pipeline.
#
#   make setup      install python + node dependencies
#   make data       build/refresh the corpus and splits
#   make pipeline   run every experiment stage end to end
#   make dev        run the API and the UI together
#   make test       run the python test suite
#
# Individual stages are also available (baselines, train, evaluate, ...) so a
# single phase can be rerun without repeating the expensive ones.

PY      ?= python
CONFIG  ?= configs/default.yaml
PORT    ?= 8000

.DEFAULT_GOAL := help
.PHONY: help setup setup-frontend data baselines train evaluate faithfulness stability \
        plausibility figures pipeline api web dev test lint clean clean-results docker

help:
	@echo "MedXplain targets:"
	@echo "  setup        install python and node dependencies"
	@echo "  data         generate the corpus, clean it and write the splits"
	@echo "  baselines    train TF-IDF + LR / RF / XGBoost"
	@echo "  train        fine-tune every transformer backbone across all seeds"
	@echo "  evaluate     metrics, calibration and significance tests"
	@echo "  faithfulness comprehensiveness / sufficiency / deletion curves"
	@echo "  stability    explanation agreement under paraphrase"
	@echo "  plausibility gold-rationale scoring + human review packet"
	@echo "  figures      all figures and LaTeX tables"
	@echo "  pipeline     every stage above, in order"
	@echo "  api / web    run one service;  dev runs both"
	@echo "  test         pytest"
	@echo "  docker       docker compose up --build"

setup:
	$(PY) -m pip install -r backend/requirements.txt
	@$(MAKE) setup-frontend

setup-frontend:
	cd frontend && npm install

data:
	$(PY) scripts/00_build_dataset.py --config $(CONFIG)

baselines:
	$(PY) scripts/01_train_baselines.py --config $(CONFIG)

train:
	$(PY) scripts/02_train_transformers.py --config $(CONFIG) --resume

evaluate:
	$(PY) scripts/03_evaluate.py --config $(CONFIG)

faithfulness:
	$(PY) scripts/04_faithfulness.py --config $(CONFIG)

stability:
	$(PY) scripts/05_stability.py --config $(CONFIG)

plausibility:
	$(PY) scripts/06_plausibility.py --config $(CONFIG)

figures:
	$(PY) scripts/07_figures_tables.py --config $(CONFIG)

pipeline: data baselines train evaluate faithfulness stability plausibility figures
	@echo "pipeline complete - see results/metrics, results/figures, results/tables"

api:
	$(PY) -m uvicorn backend.app.main:app --reload --port $(PORT)

web:
	cd frontend && npm run dev

# Runs both services; Ctrl-C stops the pair.
dev:
	@echo "API on :$(PORT), UI on :3000"
	@$(PY) -m uvicorn backend.app.main:app --port $(PORT) & \
	 cd frontend && npm run dev; \
	 kill %1 2>/dev/null || true

test:
	$(PY) -m pytest backend/tests -q

lint:
	cd frontend && npx tsc --noEmit

docker:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache frontend/.next

# Deletes trained artefacts but keeps metrics/figures/tables.
clean-results:
	rm -rf results/checkpoints results/predictions
