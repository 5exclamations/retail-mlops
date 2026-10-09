import numpy as np
import pandas as pd
import pytest

from retail_mlops.config import ForecastCfg
from retail_mlops.data.preprocess import preprocess
from retail_mlops.features.engineering import build_features, feature_columns, split_at, training_frame

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def panel(raw):
    return preprocess(raw)[0]


def test_lags_shorter_than_horizon_rejected():
    with pytest.raises(ValueError):
        ForecastCfg(horizon=7, lags=[1, 7])


def test_lag_values_correct(panel, smoke_cfg):
    f = build_features(panel, smoke_cfg.forecast)
    g = f[(f.store_id == "S01") & (f.sku_id == "K001")].reset_index(drop=True)
    assert g.loc[50, "lag_7"] == g.loc[43, "units"]
    assert g.loc[50, "lag_28"] == g.loc[22, "units"]


def test_no_leakage_from_recent_or_future_units(panel, smoke_cfg):
    """Changing units inside the last `horizon` days before t (and t itself, and after) must not change row t."""
    h = smoke_cfg.forecast.horizon
    base = build_features(panel, smoke_cfg.forecast)
    t = pd.Timestamp("2023-08-01")
    altered = panel.copy()
    altered.loc[altered["date"] > t - pd.Timedelta(days=h), "units"] = 9999.0
    f2 = build_features(altered, smoke_cfg.forecast)
    cols = feature_columns(smoke_cfg.forecast)
    a = base[base.date == t][cols].reset_index(drop=True)
    b = f2[f2.date == t][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_serving_parity_with_truncated_history(panel, smoke_cfg):
    """Features for a date computed with history cut at t-1 and units unknown for t equal the training features."""
    t = pd.Timestamp("2023-09-01")
    full = build_features(panel, smoke_cfg.forecast)
    cut = panel[panel["date"] <= t].copy()
    cut.loc[cut["date"] == t, "units"] = np.nan
    f2 = build_features(cut, smoke_cfg.forecast)
    cols = [c for c in feature_columns(smoke_cfg.forecast)]
    a = full[full.date == t][cols].reset_index(drop=True)
    b = f2[f2.date == t][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_gaps_rejected(panel, smoke_cfg):
    with pytest.raises(ValueError):
        build_features(panel.drop(panel.index[100]), smoke_cfg.forecast)


def test_split_is_strictly_temporal(panel, smoke_cfg):
    frame = training_frame(build_features(panel, smoke_cfg.forecast), smoke_cfg.forecast)
    start = frame["date"].max() - pd.Timedelta(days=27)
    pre, test = split_at(frame, start)
    assert pre["date"].max() < start <= test["date"].min()
    assert len(pre) + len(test) == len(frame)


@pytest.mark.parametrize("feature", ["roll_mean_7", "roll_mean_28", "roll_std_28", "lag_7", "lag_mean_dow", "price_ratio"])
def test_each_rolling_feature_ignores_last_horizon_days(panel, smoke_cfg, feature):
    """Per-feature leakage check: poison units in (t-h, t] and the future; feature at t must not move."""
    h = smoke_cfg.forecast.horizon
    t = pd.Timestamp("2023-07-10")
    base = build_features(panel, smoke_cfg.forecast)
    poisoned = panel.copy()
    poisoned.loc[poisoned["date"] > t - pd.Timedelta(days=h), "units"] = 1e6
    p = build_features(poisoned, smoke_cfg.forecast)
    a, b = base[base.date == t][feature], p[p.date == t][feature]
    assert a.reset_index(drop=True).equals(b.reset_index(drop=True))


def test_features_do_change_when_old_history_changes(panel, smoke_cfg):
    """Guard against a vacuous leakage test: features must depend on legitimately available history."""
    t = pd.Timestamp("2023-07-10")
    altered = panel.copy()
    altered.loc[altered["date"] == t - pd.Timedelta(days=7), "units"] += 100
    a = build_features(panel, smoke_cfg.forecast)
    b = build_features(altered, smoke_cfg.forecast)
    assert not a[a.date == t]["lag_7"].reset_index(drop=True).equals(b[b.date == t]["lag_7"].reset_index(drop=True))
