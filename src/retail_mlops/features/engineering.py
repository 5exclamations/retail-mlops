"""Feature engineering shared verbatim by training, batch scoring and the online API.

Forecast contract: a row for date *t* may only use (a) calendar/price/promo information for *t* (known in
advance) and (b) sales observed on or before *t - horizon*. Every lag is >= horizon and every rolling window is
shifted by horizon, so one model serves any lead time from 1..horizon days without leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from retail_mlops.calendar import holiday_frame
from retail_mlops.config import ForecastCfg

SERIES = ["store_id", "sku_id"]
CATEGORICAL = ["store_id", "sku_id", "category"]
TARGET = "units"


def feature_columns(cfg: ForecastCfg) -> list[str]:
    cols = list(CATEGORICAL)
    cols += ["dow", "dom", "month", "week_of_year", "is_weekend", "doy_sin", "doy_cos"]
    cols += ["is_holiday", "is_black_friday", "days_to_holiday", "price", "price_ratio", "promo"]
    cols += [f"lag_{k}" for k in cfg.lags] + [f"promo_lag_{k}" for k in cfg.lags] + ["lag_mean_dow"]
    cols += [f"roll_mean_{w}" for w in cfg.rolling_windows] + ["roll_std_28"]
    return cols


def numeric_columns(cfg: ForecastCfg) -> list[str]:
    return [c for c in feature_columns(cfg) if c not in CATEGORICAL]


def _assert_gap_free(df: pd.DataFrame) -> None:
    gaps = df.groupby(SERIES)["date"].diff().dropna()
    if (gaps != pd.Timedelta(days=1)).any():
        raise ValueError("panel has date gaps or duplicates; run preprocess.complete_panel first")


def build_features(df: pd.DataFrame, cfg: ForecastCfg) -> pd.DataFrame:
    """Input: gap-free panel (date, store_id, sku_id, category, units, price, promo); units may be NaN for the
    rows to be forecast. Output: the panel plus feature columns (row order preserved per series/date)."""
    df = df.sort_values(SERIES + ["date"]).reset_index(drop=True)
    _assert_gap_free(df)
    h = cfg.horizon
    out = df.copy()
    g_units = out.groupby(SERIES)["units"]
    g_promo = out.groupby(SERIES)["promo"]

    d = out["date"]
    out["dow"] = d.dt.dayofweek
    out["dom"] = d.dt.day
    out["month"] = d.dt.month
    out["week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    doy = d.dt.dayofyear.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out = pd.concat([out, holiday_frame(d).reset_index(drop=True)], axis=1)

    for k in cfg.lags:
        out[f"lag_{k}"] = g_units.shift(k)
        out[f"promo_lag_{k}"] = g_promo.shift(k)
    out["lag_mean_dow"] = out[[f"lag_{k}" for k in cfg.lags]].mean(axis=1)

    shifted = g_units.shift(h)
    shifted_grp = shifted.groupby([out["store_id"], out["sku_id"]])
    for w in cfg.rolling_windows:
        out[f"roll_mean_{w}"] = shifted_grp.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    out["roll_std_28"] = shifted_grp.transform(lambda s: s.rolling(28, min_periods=2).std())

    ref_price = out.groupby(SERIES)["price"].shift(h)
    ref_price = ref_price.groupby([out["store_id"], out["sku_id"]]).transform(
        lambda s: s.rolling(28, min_periods=1).mean()
    )
    out["price_ratio"] = out["price"] / ref_price - 1.0
    return out


def training_frame(feats: pd.DataFrame, cfg: ForecastCfg) -> pd.DataFrame:
    """Drop warm-up rows (no full lag history) and rows without a target."""
    need = [f"lag_{max(cfg.lags)}", "roll_std_28", "price_ratio", TARGET]
    return feats.dropna(subset=need).reset_index(drop=True)


def split_at(frame: pd.DataFrame, test_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict temporal split: everything before test_start is available for fitting/selection, the rest is the
    final held-out window."""
    return frame[frame["date"] < test_start], frame[frame["date"] >= test_start]
