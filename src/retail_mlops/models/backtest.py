"""Walk-forward (expanding window) backtesting. Folds sit immediately BEFORE the final test window, so model
selection never touches the test data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from retail_mlops.features.engineering import TARGET
from retail_mlops.models.metrics import compute_metrics


def fold_bounds(test_start: pd.Timestamp, folds: int, step_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [(test_start - pd.Timedelta(days=(folds - i) * step_days),
             test_start - pd.Timedelta(days=(folds - i - 1) * step_days)) for i in range(folds)]


def walk_forward(frame: pd.DataFrame, cols: list[str], estimator, test_start: pd.Timestamp, folds: int,
                 step_days: int) -> dict:
    """Fit on everything before each fold, score the fold. Returns per-fold, pooled and per-lead metrics.

    `by_lead` groups days by position inside the forecast block (1..7, repeating weekly). The model's inputs do
    not change with lead (all lags >= horizon), so this is a sanity check that error does not depend on lead,
    not a lead-time accuracy curve."""
    out_folds, pooled_y, pooled_p, leads = [], [], [], []
    for start, end in fold_bounds(test_start, folds, step_days):
        train = frame[frame["date"] < start]
        test = frame[(frame["date"] >= start) & (frame["date"] < end)]
        if train.empty or test.empty:
            continue
        assert train["date"].max() < test["date"].min(), "walk-forward fold overlaps in time"
        est = clone(estimator).fit(train[cols], train[TARGET])
        pred = np.clip(est.predict(test[cols]), 0, None)
        out_folds.append({"start": str(start.date()), "end": str((end - pd.Timedelta(days=1)).date()),
                          "n_train": int(len(train)), **compute_metrics(test[TARGET], pred)})
        pooled_y.append(test[TARGET].to_numpy())
        pooled_p.append(pred)
        leads.append(((test["date"] - start).dt.days % 7 + 1).to_numpy())
    if not out_folds:
        raise ValueError("no usable backtest folds; not enough history before the test window")
    y, p, ld = np.concatenate(pooled_y), np.concatenate(pooled_p), np.concatenate(leads)
    wapes = [f["wape"] for f in out_folds]
    return {"folds": out_folds, "mean_wape": float(np.mean(wapes)), "std_wape": float(np.std(wapes)),
            "pooled": compute_metrics(y, p),
            "by_lead": {int(k): compute_metrics(y[ld == k], p[ld == k]) for k in np.unique(ld)}}
