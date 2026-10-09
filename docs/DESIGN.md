# Design notes

## 1. ML lifecycle
`generate/ingest → validate → preprocess → features → train (baselines + candidates) → evaluate on untouched test
window → promotion gates → register (MLflow) → champion alias → serve → monitor → retrain trigger → (loop)`.
Everything is a CLI command (`python -m retail_mlops …`) and each step is a plain function, so CI, cron or an
orchestrator can call it.

## 2. Data flow
```
synthetic generator ──raw CSV──► validate ──► preprocess ──► sales_history (SQL: Postgres/SQLite)
                                                                 │  ├─► training (read_sales → validate → …)
                                          /ground-truth ─────────┘  └─► feature source for /predict + batch
/predict, batch ─► prediction_log (features JSON, model version) ─► drift + live accuracy (joined with actuals)
```
Tables: `sales_history` (history + ground truth), `prediction_log`, `retrain_runs`.
Raw-data issues are injected on purpose (duplicates, missing price/promo); preprocessing dedupes (later row wins),
completes the daily panel, forward-fills price, treats unknown promo as 0, clips negative units.

## 3. Model architecture
Task: forecast daily units for a store×SKU up to **7 days ahead** with one model.
Leak-free contract: calendar/price/promo of the target day are inputs (known in advance); sales are only used with
lag ≥ 7 days (lags 7/14/21/28, rolling mean 7/28, rolling std 28, promo at lagged days, price vs 28-day reference).
`ForecastCfg` refuses lags shorter than the horizon, and tests check that altering recent units cannot change a
row's features and that serving-time features equal training features.
Models: seasonal-naive and moving-average baselines; Ridge (one-hot + scaled numerics); HistGradientBoosting with
Poisson loss and native categorical handling. Selection on the validation window; the selected model is the
one evaluated on test and shipped (not refit on train+valid, so what is measured is what is served).
Primary metric WAPE; also MAE, RMSE, relative bias.

## 4. Experiment tracking
One MLflow run per training call with nested runs per model. Logged: flattened config, data hash, git SHA,
Python/sklearn versions, split dates and sizes, validation + preprocessing reports, per-model valid/test metrics,
gate results, per-category metrics, the model, and a `bundle/` (exact training config, metadata, monitoring
reference sample + reference predictions). Registry: model `retail-demand-forecaster`, versions carry tags;
aliases `candidate`, `champion`, `previous`.

**Promotion gates** (`registry/gates.py`, thresholds in config): data validation passed; ≥3% relative WAPE
improvement over best baseline; |bias| ≤ 5%; no category >5% worse than baseline; not worse than the current
champion **re-scored on the same new test window**. `rollback` points `champion` back to `previous`; the API picks
this up on its next alias check.

## 5. Deployment architecture
FastAPI (`/predict`, `/ground-truth`, `/model`, `/ready`, `/health`, `/metrics`, `/monitoring/*`, `/admin/reload`)
loads `models:/…@champion` and builds features from SQL history using the same code as training. Batch scoring
(`batch-predict`) uses the same predictor. docker-compose: Postgres, MLflow server (Postgres backend, proxied
artifacts), API, Prometheus; one image for all. Admin/ground-truth endpoints accept an `X-API-Key` when
`ADMIN_API_KEY` is set. Models are stored with cloudpickle (MLflow's default skops refuses gradient-boosting
trees); only load models from a registry you control.

## 6. Monitoring
Prometheus: request count/latency, model-inference latency, prediction counts and value histogram, error counters
by type, per-feature PSI and alert flags, drift share, prediction PSI, live WAPE/MAE/bias, ground-truth coverage,
degradation flag, served-model info. Rules in `monitoring/alerts.yml`.
- *Input drift*: PSI (reference-quantile bins) + KS test on logged inputs vs. the training reference; "alert" needs
  PSI ≥ 0.25 **and** KS p < 0.001; overall drift = ≥25% of monitored features in alert. Calendar features are not
  monitored (they shift legitimately with the season).
- *Prediction drift*: PSI/KS of predictions vs. reference predictions.
- *Performance*: predictions joined to actuals as they arrive (`/ground-truth` or `ingest`); degraded if live
  WAPE ≥ 1.25× the training test WAPE.
A background task refreshes these every `refresh_seconds` and re-checks the champion alias.

## 7. Reproducibility
Seeded generator and models; pinned `requirements.txt` (+ full `requirements.lock`); data content hash, git SHA and
config stored on every run; the serving process uses the config the model was trained with. Tests assert generator
determinism. Same-machine reruns should reproduce metrics; cross-platform bit-identity is not claimed.

## 8. Limitations
- Synthetic data: the generator's structure (weekly/yearly seasonality, promo lift, price elasticity) is exactly what
  the features target, so absolute accuracy says nothing about real retail. Stock-outs are recorded as zero sales
  (censored demand is not modelled).
- Point forecasts only (no prediction intervals); horizon fixed at 7 days; no new-product (cold-start) handling –
  unknown series return 422, short histories are flagged `low_history`.
- Planned future prices default to the last known price; promos default to none unless supplied.
- Retrain gate tie-handling: a candidate within noise of the champion is promoted (observed: v2 vs v1, 0.3016 vs
  0.3017). The champion may also have trained on part of the new test window, which flatters it; a minimum-improvement
  margin or significance test would be better.
- The drift reference mixes all seasons, so seasonal swings can look like drift; thresholds are untuned defaults.
  With a drift scenario of this size the monitors flagged only warnings, not an automatic retrain.
- Live accuracy only covers predictions whose actuals have arrived; coverage is exported.
- Not done: authentication beyond a shared API key, multi-replica model reload coordination, load testing,
  CI/Docker execution (see README "Status").
