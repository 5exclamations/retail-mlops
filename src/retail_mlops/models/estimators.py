"""Baselines and candidate estimators. All take a DataFrame with the columns from feature_columns()."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from retail_mlops.config import Config
from retail_mlops.features.engineering import CATEGORICAL, numeric_columns


class ColumnBaseline(BaseEstimator, RegressorMixin):
    """Predict directly from a (lagged) feature column, falling back to another column when it is missing."""

    def __init__(self, column: str = "lag_7", fallback: str = "roll_mean_28"):
        self.column = column
        self.fallback = fallback

    def fit(self, X, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].fillna(X[self.fallback]).fillna(0.0).to_numpy(dtype=float)


def make_baselines(cfg: Config) -> dict[str, ColumnBaseline]:
    longest = max(cfg.forecast.rolling_windows)
    return {
        "seasonal_naive": ColumnBaseline("lag_7", f"roll_mean_{longest}"),  # same weekday last week
        "moving_average": ColumnBaseline(f"roll_mean_{longest}", f"roll_mean_{min(cfg.forecast.rolling_windows)}"),
    }


def make_ridge(cfg: Config) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         numeric_columns(cfg.forecast)),
    ])
    return Pipeline([("pre", pre), ("model", Ridge(alpha=cfg.models.ridge_alpha))])


def make_hgb(cfg: Config) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CATEGORICAL),
            ("num", "passthrough", numeric_columns(cfg.forecast)),
        ],
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(
        loss="poisson",  # demand is a non-negative count
        categorical_features=list(range(len(CATEGORICAL))),
        early_stopping=False,  # sklearn's early stopping uses a random (leaky) validation split
        random_state=cfg.seed,
        **cfg.models.hgb,
    )
    return Pipeline([("pre", pre), ("model", model)])


def make_candidates(cfg: Config) -> dict[str, Pipeline]:
    return {"ridge": make_ridge(cfg), "hist_gradient_boosting": make_hgb(cfg)}
