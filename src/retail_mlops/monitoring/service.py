"""Builds the combined monitoring report (input drift, prediction drift, live performance)."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy.engine import Engine

from retail_mlops.config import MonitoringCfg
from retail_mlops.db import read_prediction_log
from retail_mlops.features.engineering import CATEGORICAL
from retail_mlops.monitoring.drift import drift_report, prediction_drift
from retail_mlops.monitoring.performance import performance_report
from retail_mlops.registry.bundle import ModelBundle

MAX_ROWS = 20000


def _features_frame(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    df = pd.DataFrame([json.loads(s) for s in preds["features"]])
    for c in df.columns:
        if c not in CATEGORICAL:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_report(engine: Engine, bundle: ModelBundle, cfg: MonitoringCfg, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC).replace(tzinfo=None)
    since = now - timedelta(days=cfg.window_days)
    preds = read_prediction_log(engine, since=since, model_version=bundle.version)
    if len(preds) > MAX_ROWS:
        preds = preds.sample(MAX_ROWS, random_state=0)
    live = _features_frame(preds)
    report = {
        "generated_at": now.isoformat(), "model_version": bundle.version, "window_days": cfg.window_days,
        "n_predictions_in_window": int(len(preds)),
        "drift": drift_report(bundle.reference, live, cfg.drift_features, cfg) if len(live) else
        {"status": "insufficient_data", "n_current": 0, "features": [], "drift_share": 0.0, "drifted": False},
        "prediction_drift": prediction_drift(bundle.reference["__prediction"], preds["prediction"], cfg)
        if len(preds) else {"status": "insufficient_data", "psi": None},
        "performance": performance_report(engine, since, bundle.reference_wape, cfg.perf_degradation_ratio,
                                          cfg.min_ground_truth, model_version=bundle.version,
                                          maturity_coverage=cfg.maturity_coverage),
    }
    return report
