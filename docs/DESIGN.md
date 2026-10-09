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
Poisson loss and native categorical handling. **Selection uses walk-forward CV only** (folds immediately before the
test window); the selected recipe is refit on all pre-test data and evaluated once on the test window. If a champion
exists, the test window starts after the champion's last data day so neither model has seen it.
Primary metric WAPE; also MAE, RMSE, relative bias; breakdowns by category/store/SKU are logged per run.

## 4. Experiment tracking
One MLflow run per training call with nested runs per model. Logged: flattened config, data hash, git SHA,
Python/sklearn versions, split dates, validation + preprocessing reports, per-model CV/test metrics, gate results,
per-category/store/SKU breakdowns, the model (skops) and a `bundle/` (training config, metadata incl. artifact SHA-256,
monitoring reference sample). Registry: aliases `candidate`, `champion`, `previous`; versions are never deleted.

**Promotion gates** (`registry/gates.py`, thresholds in config): data validation; >=3% relative WAPE improvement over the
best baseline; |bias| <= 5%; no category >5% worse than baseline; >= `min_holdout_days` fresh days; >= 1% relative
improvement over the champion on identical fresh rows; and a paired block-bootstrap CI (resampling days) that excludes
0. Each failed check yields a reason string. See README for the full protocol and demo behaviour.

## 5. Deployment architecture
FastAPI (`/predict`, `/ground-truth`, `/model`, `/ready`, `/health`, `/metrics`, `/monitoring/*`, `/admin/reload`)
loads `models:/…@champion` and builds features from SQL history using the same code as training. Batch scoring
(`batch-predict`) uses the same predictor. docker-compose: Postgres, MLflow server (Postgres backend, proxied
artifacts), API, Prometheus; one image for all. Admin/ground-truth endpoints are disabled until `ADMIN_API_KEY` is set, then require `X-API-Key`. Models are stored in skops format with an allow-list of reviewed types and a registry-held SHA-256
verified before loading.

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
- The fresh-holdout protocol means a challenger never learns from the days used to judge it (see README trade-offs).
- The drift reference mixes all seasons, so seasonal swings can look like drift; thresholds are untuned defaults.
  With a drift scenario of this size the monitors flagged only warnings, not an automatic retrain.
- Live accuracy only covers predictions whose actuals have arrived; coverage is exported.
- Not done: authentication beyond a shared API key, multi-replica model reload coordination, load testing,
  CI/Docker execution (see README "Status").
