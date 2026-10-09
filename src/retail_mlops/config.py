"""Typed configuration: code defaults < YAML file(s) < environment variables."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class DataCfg(BaseModel):
    n_stores: int = 8
    n_skus: int = 20
    start_date: str = "2023-01-01"
    n_days: int = 730
    inject_quality_issues: bool = True
    raw_path: str = "data/raw/sales.csv"
    processed_path: str = "data/processed/features.parquet"


class ValidationCfg(BaseModel):
    max_null_fraction: float = 0.01
    max_duplicate_fraction: float = 0.01
    max_missing_day_fraction: float = 0.02
    min_history_days: int = 60
    max_units: float = 100000
    outlier_multiple_warn: float = 25.0


class ForecastCfg(BaseModel):
    horizon: int = 7
    lags: list[int] = [7, 14, 21, 28]
    rolling_windows: list[int] = [7, 28]
    valid_days: int = 56
    test_days: int = 56

    @model_validator(mode="after")
    def _lags_respect_horizon(self) -> ForecastCfg:
        bad = [lag for lag in self.lags if lag < self.horizon]
        if bad:
            raise ValueError(f"lags {bad} are shorter than horizon={self.horizon}: that would leak the future")
        return self


class ModelsCfg(BaseModel):
    ridge_alpha: float = 10.0
    hgb: dict[str, Any] = Field(
        default_factory=lambda: dict(
            learning_rate=0.08, max_iter=300, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0
        )
    )
    selection_metric: str = "wape"
    reference_sample_size: int = 20000


class GatesCfg(BaseModel):
    min_improvement_vs_baseline: float = 0.03  # relative WAPE reduction vs best baseline
    max_regression_vs_champion: float = 0.0  # relative WAPE increase tolerated vs champion (same holdout)
    max_abs_bias: float = 0.05  # |sum(pred-actual)/sum(actual)|
    max_segment_regression_vs_baseline: float = 0.05  # per-category WAPE vs baseline, relative
    require_valid_data: bool = True


class MonitoringCfg(BaseModel):
    drift_features: list[str] = [
        "lag_7", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_28", "price", "price_ratio", "promo",
    ]
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    ks_pvalue: float = 0.001
    min_samples: int = 200
    window_days: int = 14
    drift_share_alert: float = 0.25  # share of monitored features in alert -> "drifted"
    perf_degradation_ratio: float = 1.25  # live WAPE / reference test WAPE
    min_ground_truth: int = 100
    refresh_seconds: int = 300
    reload_check_seconds: int = 60


class RetrainCfg(BaseModel):
    max_model_age_days: int = 30
    min_new_rows: int = 5000
    training_window_days: int | None = None  # None = use all history


class MlflowCfg(BaseModel):
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment: str = "retail-demand-forecasting"
    model_name: str = "retail-demand-forecaster"
    champion_alias: str = "champion"
    candidate_alias: str = "candidate"
    previous_alias: str = "previous"


class Config(BaseModel):
    seed: int = 42
    database_url: str = "sqlite:///retail.db"
    data: DataCfg = DataCfg()
    validation: ValidationCfg = ValidationCfg()
    forecast: ForecastCfg = ForecastCfg()
    models: ModelsCfg = ModelsCfg()
    gates: GatesCfg = GatesCfg()
    monitoring: MonitoringCfg = MonitoringCfg()
    retrain: RetrainCfg = RetrainCfg()
    mlflow: MlflowCfg = MlflowCfg()


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(*paths: str | Path | None, overrides: dict | None = None) -> Config:
    """Merge YAML files in order, then explicit overrides, then env vars."""
    merged: dict = {}
    files = [p for p in paths if p] or ([DEFAULT_CONFIG_PATH] if DEFAULT_CONFIG_PATH.exists() else [])
    for p in files:
        merged = _deep_merge(merged, yaml.safe_load(Path(p).read_text()) or {})
    if overrides:
        merged = _deep_merge(merged, overrides)
    if os.getenv("DATABASE_URL"):
        merged["database_url"] = os.environ["DATABASE_URL"]
    if os.getenv("MLFLOW_TRACKING_URI"):
        merged.setdefault("mlflow", {})["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    return Config(**merged)
