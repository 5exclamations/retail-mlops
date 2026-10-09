"""Live model performance from ground truth: join logged predictions with actuals that arrived later."""
from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from retail_mlops.db import read_prediction_log, read_sales
from retail_mlops.models.metrics import compute_metrics


def join_with_actuals(engine: Engine, preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return preds.assign(actual=[])
    actuals = read_sales(engine, start=preds["target_date"].min(), end=preds["target_date"].max())
    actuals = actuals[["store_id", "sku_id", "date", "units", "category"]].rename(columns={"date": "target_date", "units": "actual"})
    # if the same (series, date) was predicted several times keep the latest prediction
    preds = preds.sort_values("created_at").drop_duplicates(["store_id", "sku_id", "target_date", "model_version"], keep="last")
    return preds.merge(actuals, on=["store_id", "sku_id", "target_date"], how="left")


def performance_report(engine: Engine, since, reference_wape: float | None, degradation_ratio: float,
                       min_ground_truth: int, model_version: str | None = None) -> dict:
    preds = read_prediction_log(engine, since=since, model_version=model_version)
    joined = join_with_actuals(engine, preds)
    n_pred = len(joined)
    labelled = joined.dropna(subset=["actual"]) if n_pred else joined
    rep = {"n_predictions": int(n_pred), "n_with_ground_truth": int(len(labelled)),
           "coverage": float(len(labelled) / n_pred) if n_pred else 0.0}
    if len(labelled) < min_ground_truth:
        return {**rep, "status": "insufficient_ground_truth", "min_ground_truth": min_ground_truth}
    m = compute_metrics(labelled["actual"], labelled["prediction"])
    rep.update(status="ok", **{k: m[k] for k in ("mae", "rmse", "wape", "bias")})
    rep["by_category"] = {c: compute_metrics(g["actual"], g["prediction"])
                          for c, g in labelled.groupby("category")}
    if reference_wape:
        rep["reference_wape"] = reference_wape
        rep["wape_ratio"] = m["wape"] / reference_wape
        rep["degraded"] = bool(rep["wape_ratio"] >= degradation_ratio)
    return rep
