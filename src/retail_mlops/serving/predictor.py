"""Online/batch scoring path. Builds features from the SQL history with the same code as training."""
from __future__ import annotations

import json
import math
import uuid

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from retail_mlops.data.preprocess import complete_panel
from retail_mlops.db import read_sales
from retail_mlops.features.engineering import build_features
from retail_mlops.registry.bundle import ModelBundle

LOOKBACK_DAYS = 70  # >= max lag (28) + horizon + 28-day rolling window slack


class ForecastError(ValueError):
    """Request cannot be scored; `problems` is a list of human-readable reasons."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer,)) else v


def forecast(engine: Engine, bundle: ModelBundle, instances: list[dict], source: str = "api",
             request_id: str | None = None) -> tuple[list[dict], list[dict]]:
    """Returns (results, prediction_log_rows). Raises ForecastError for unscorable requests."""
    request_id = request_id or uuid.uuid4().hex[:16]
    h = bundle.cfg.forecast.horizon
    req = pd.DataFrame(instances)
    req["date"] = pd.to_datetime(req["date"]).dt.normalize()
    for c in ("price", "promo"):
        if c not in req:
            req[c] = np.nan
    req = req.reset_index(drop=True)
    req["_rid"] = np.arange(len(req))

    pairs = list(req[["store_id", "sku_id"]].drop_duplicates().itertuples(index=False, name=None))
    start = req["date"].min() - pd.Timedelta(days=LOOKBACK_DAYS)
    hist = read_sales(engine, start=start, end=req["date"].max(), series=pairs)

    problems = []
    last_seen = hist.groupby(["store_id", "sku_id"])["date"].max().to_dict()
    cats = hist.groupby(["store_id", "sku_id"])["category"].first().to_dict()
    for p in pairs:
        if p not in last_seen:
            problems.append(f"{p[0]}/{p[1]}: no sales history within {LOOKBACK_DAYS} days before the requested date")
    if problems:
        raise ForecastError(problems)
    for r in req.itertuples():
        gap = (r.date - last_seen[(r.store_id, r.sku_id)]).days
        if gap > h:
            problems.append(f"{r.store_id}/{r.sku_id} {r.date.date()}: latest history is "
                            f"{last_seen[(r.store_id, r.sku_id)].date()}; max forecast lead is {h} days")
    if problems:
        raise ForecastError(sorted(set(problems)))

    req["category"] = [cats[(s, k)] for s, k in zip(req["store_id"], req["sku_id"], strict=True)]
    # last known price per series is the default for unplanned prices
    last_price = hist.sort_values("date").groupby(["store_id", "sku_id"])["price"].last().to_dict()
    req["price"] = [p if pd.notna(p) else last_price[(s, k)]
                    for p, s, k in zip(req["price"], req["store_id"], req["sku_id"], strict=True)]
    req["promo"] = req["promo"].fillna(0).astype(int)

    # history rows win for units (observed), request rows supply planned price/promo
    ov = req[["store_id", "sku_id", "date", "price", "promo", "category"]].drop_duplicates(["store_id", "sku_id", "date"], keep="last")
    base = hist.merge(ov[["store_id", "sku_id", "date", "price", "promo"]], on=["store_id", "sku_id", "date"],
                      how="outer", suffixes=("", "_req"))
    for c in ("price", "promo"):
        base[c] = base[f"{c}_req"].combine_first(base[c])
    base = base.drop(columns=["price_req", "promo_req"])
    base["category"] = [cats.get((s, k)) for s, k in zip(base["store_id"], base["sku_id"], strict=True)]
    panel = complete_panel(base.sort_values(["store_id", "sku_id", "date"]), end_date=req["date"].max())
    panel["price"] = panel.groupby(["store_id", "sku_id"])["price"].transform(lambda s: s.ffill().bfill())

    feats = build_features(panel, bundle.cfg.forecast)
    keyed = req.merge(feats.drop(columns=["category", "price", "promo", "units"]), on=["store_id", "sku_id", "date"], how="left")
    keyed["category"] = req["category"].to_numpy()
    keyed["price"] = req["price"].to_numpy()
    keyed["promo"] = req["promo"].to_numpy()
    preds = bundle.predict_features(keyed)
    low_history = keyed["lag_28"].isna().to_numpy()

    results, rows = [], []
    for i, r in enumerate(keyed.itertuples()):
        results.append({"store_id": r.store_id, "sku_id": r.sku_id, "date": r.date.date().isoformat(),
                        "prediction": round(float(preds[i]), 3), "low_history": bool(low_history[i]),
                        "model_version": bundle.version})
        rows.append({"request_id": request_id, "source": source, "model_name": bundle.name,
                     "model_version": bundle.version, "store_id": r.store_id, "sku_id": r.sku_id,
                     "target_date": r.date.date(), "prediction": float(preds[i]),
                     "features": json.dumps({c: _clean(keyed.iloc[i][c]) for c in bundle.columns})})
    return results, rows
