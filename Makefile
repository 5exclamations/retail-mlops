PY ?= .venv/bin/python
export MLFLOW_DISABLE_AGENT_HINT=1

install:      ## create venv + install
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -r requirements.txt pytest pytest-cov httpx ruff && uv pip install --python $(PY) --no-deps -e .
lint:
	.venv/bin/ruff check src tests
test:
	$(PY) -m pytest -q
data:
	$(PY) -m retail_mlops generate && $(PY) -m retail_mlops validate
train:
	$(PY) -m retail_mlops train
smoke:        ## what CI runs
	$(PY) -m retail_mlops --config configs/smoke.yaml generate && $(PY) -m retail_mlops --config configs/smoke.yaml validate && $(PY) -m retail_mlops --config configs/smoke.yaml train --no-register
demo:         ## full local loop on SQLite: train -> serve data -> drift -> monitor -> retrain
	rm -f retail.db mlflow.db && $(PY) -m retail_mlops generate && $(PY) -m retail_mlops train && \
	$(PY) -m retail_mlops ingest --end 2024-12-30 && ./scripts/demo_drift.sh && \
	$(PY) -m retail_mlops monitor && $(PY) -m retail_mlops retrain --dry-run && $(PY) -m retail_mlops retrain --force
serve:
	$(PY) -m uvicorn retail_mlops.serving.app:create_app --factory --port 8000
stack:
	docker compose up -d --build
.PHONY: install lint test data train smoke demo serve stack
