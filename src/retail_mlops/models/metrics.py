from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    err = p - y
    total = float(np.abs(y).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "wape": float(np.abs(err).sum() / total) if total > 0 else float("nan"),  # primary metric
        "bias": float(err.sum() / total) if total > 0 else float("nan"),  # >0 = over-forecast
        "n": int(len(y)),
    }


def segment_metrics(df: pd.DataFrame, pred: np.ndarray, by: str) -> dict[str, dict[str, float]]:
    tmp = pd.DataFrame({by: df[by].to_numpy(), "y": df["units"].to_numpy(), "p": np.asarray(pred)})
    return {str(k): compute_metrics(g["y"], g["p"]) for k, g in tmp.groupby(by)}
