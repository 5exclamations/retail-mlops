"""Live model performance from ground truth.

Only *matured* forecasts are scored: a prediction counts once its target date has complete actuals (>= maturity
coverage of the known series have a units value) and it was a genuine forecast (horizon >= 1, i.e. the actual did
not exist when the forecast was issued). Everything else is reported as pending/retrospective, never as error."""
from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from retail_mlops.db import read_prediction_log, read_sales
from retail_mlops.models.metrics import compute_metrics


def _breakdown(df: pd.DataFrame, by: str) -> dict:
    return {str(k): compute_metrics(g["actual"], g["prediction"]) for k, g in df.groupby(by)}


def join_with_actuals(engine: Engine, preds: pd.DataFrame, maturity_coverage: float = 0.95) -> pd.DataFrame:
    """Adds `actual` (NaN unless the date is matured) and `status` in {scored, pending, retrospective}."""
    if preds.empty:
        return preds.assign(actual=[], status=[])
    actuals = read_sales(engine, start=preds["target_date"].min(), end=preds["target_date"].max())
    n_series = max(1, len(read_sales(engine, start=preds["target_date"].min() - pd.Timedelta(days=70),
                                     end=preds["target_date"].max())[["store_id", "sku_id"]].drop_duplicates()))
    have = actuals.dropna(subset=["units"]).groupby("date").size()
    matured = set(have[have / n_series >= maturity_coverage].index)
    actuals = actuals[["store_id", "sku_id", "date", "units", "category"]].rename(
        columns={"date": "target_date", "units": "actual"})
    # one score per (series, date, model version, horizon): repeated identical requests are not extra evidence
    preds = preds.sort_values("created_at").drop_duplicates(
        ["store_id", "sku_id", "target_date", "model_version", "horizon"], keep="last")
    out = preds.merge(actuals, on=["store_id", "sku_id", "target_date"], how="left")
    is_matured = out["target_date"].isin(matured)
    retro = out["horizon"].fillna(1) < 1
    out.loc[~is_matured | retro, "actual"] = float("nan")
    out["status"] = "scored"
    out.loc[~is_matured, "status"] = "pending"
    out.loc[retro, "status"] = "retrospective"
    out.loc[out["status"] == "scored", "status"] = out["actual"].notna().map({True: "scored", False: "pending"})
    return out


def performance_report(engine: Engine, since, reference_wape: float | None, degradation_ratio: float,
                       min_ground_truth: int, model_version: str | None = None,
                       maturity_coverage: float = 0.95) -> dict:
    preds = read_prediction_log(engine, since=since, model_version=model_version)
    joined = join_with_actuals(engine, preds, maturity_coverage)
    n_pred = len(joined)
    scored = joined[joined["status"] == "scored"] if n_pred else joined
    rep = {"n_predictions": int(n_pred), "n_with_ground_truth": int(len(scored)),
           "n_pending": int((joined["status"] == "pending").sum()) if n_pred else 0,
           "n_retrospective": int((joined["status"] == "retrospective").sum()) if n_pred else 0,
           "coverage": float(len(scored) / n_pred) if n_pred else 0.0}
    if len(scored) < min_ground_truth:
        return {**rep, "status": "insufficient_ground_truth", "min_ground_truth": min_ground_truth}
    m = compute_metrics(scored["actual"], scored["prediction"])
    rep.update(status="ok", **{k: m[k] for k in ("mae", "rmse", "wape", "bias")})
    rep["by_category"] = _breakdown(scored, "category")
    rep["by_store"] = _breakdown(scored, "store_id")
    rep["by_sku"] = _breakdown(scored, "sku_id")
    rep["by_horizon"] = _breakdown(scored, "horizon")
    if reference_wape:
        rep["reference_wape"] = reference_wape
        rep["wape_ratio"] = m["wape"] / reference_wape
        rep["degraded"] = bool(rep["wape_ratio"] >= degradation_ratio)
    return rep
