# Retail demand forecasting – MLOps platform

An end-to-end, CPU-only MLOps system around a store × SKU × day demand forecaster: synthetic data → validation →
preprocessing → features → baselines + candidate models → MLflow tracking/registry → automated promotion gates →
FastAPI service + batch scoring → Prometheus monitoring (latency, drift, prediction distribution, errors, live
accuracy from ground truth) → drift/performance/schedule-triggered retraining → CI.

**All data is synthetic** (`src/retail_mlops/data/generate.py`). Nothing here is evidence about real retail demand.

## Quick start (no Docker, SQLite, ~5 minutes)

```bash
make install        # uv venv (Python 3.12) + dependencies
make test           # 34 tests
make data train     # generate 8 stores x 20 SKUs x 730 days, validate, train, gate, register -> champion
python -m retail_mlops ingest --end 2024-12-30      # load history into the DB (sales_history)
python -m retail_mlops batch-predict --output data/out/forecast.csv
make serve          # http://localhost:8000/docs   (metrics: /metrics)
make demo           # whole loop incl. simulated drift, monitoring and a forced retrain
```

Full stack (Postgres, MLflow server, API, Prometheus): `docker compose up -d --build`
(MLflow UI :5000, API :8000, Prometheus :9090). **The compose/Dockerfile were written but could not be built in the
authoring environment (host disk full), so they are untested** – see "Status".

CLI: `python -m retail_mlops {generate,validate,ingest,train,batch-predict,monitor,retrain,rollback}`.

## Results actually obtained (seed 42, this machine; reproduce with `make data train`)

Held-out test window = last 56 days; forecasts use information available 7 days earlier.

| model | test WAPE | note |
|---|---|---|
| seasonal naive (same weekday last week) | – | baselines are compared on validation; best baseline test WAPE = **0.3925** |
| **hist. gradient boosting (Poisson loss), selected** | **0.2877** | MAE 11.93, RMSE 17.73, bias −1.3% |

The candidate beat the best baseline by 26.7% (relative WAPE) and passed all gates; training takes ~42 s on CPU.
On the 300-day smoke dataset the same code **rejected** its candidate (|bias| 10% > 5% gate) – the gate working as
intended, and a reminder that small data gives unreliable models.

Drift demo (4 weeks after training, synthetic shift from 2024-12-31: beverages +35%, snacks −20%, promo lift halved,
prices +10%): input drift was *mild* – only `price_ratio` hit "alert" (1 of 8 features, below the 25% share needed to
flag drift), prediction PSI 0.196 ("warn"), live WAPE 0.349 vs 0.288 reference (1.21×, below the 1.25× trigger).
So no automatic retrain fired (`retrain --dry-run` → skipped). A **forced** retrain produced v2, promoted with test
WAPE 0.3016 vs the champion's 0.3017 on the same window – i.e. no meaningful gain (see limitations).

## Documentation
[docs/DESIGN.md](docs/DESIGN.md): ML lifecycle, data flow, model architecture, experiment tracking, deployment,
monitoring, reproducibility, limitations.

## Status
| area | state |
|---|---|
| data/features/models/registry/gates/API/batch/monitoring/retrain | implemented, exercised |
| tests | 34 pass on SQLite; the 8 integration/API tests also pass on Postgres 17 |
| CI workflow (`.github/workflows/ci.yml`) | written, **never executed** on GitHub |
| Dockerfile / docker-compose / Prometheus rules | written, **not built or run** |
| API load/latency benchmarks | not done (one local `/predict` of 1 item took ~0.5 s incl. warm-up) |
