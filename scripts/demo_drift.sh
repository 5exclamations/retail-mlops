#!/usr/bin/env bash
# Simulates 4 weeks of live operation after the initial model, with a demand/price/promo shift starting 2024-12-31.
set -euo pipefail
PY=${PY:-.venv/bin/python}; export MLFLOW_DISABLE_AGENT_HINT=1
$PY -m retail_mlops generate --days 814 --drift-start 2024-12-31 --out data/raw/future.csv
for end in 2025-01-06 2025-01-13 2025-01-20 2025-01-27; do
  $PY -m retail_mlops ingest --input data/raw/future.csv --start 2024-12-31 --end $end | tail -1 | cut -c1-80
  $PY -m retail_mlops batch-predict --output data/out/forecast_$end.csv | tail -1
done
