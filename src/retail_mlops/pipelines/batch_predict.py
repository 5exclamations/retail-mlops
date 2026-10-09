"""Batch scoring: forecast the next `days` days for every store x SKU and write CSV + prediction log."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from retail_mlops.db import last_sales_date, log_predictions, read_sales
from retail_mlops.registry.bundle import ModelBundle
from retail_mlops.serving.predictor import LOOKBACK_DAYS, forecast

CHUNK = 2000


def run_batch(engine: Engine, bundle: ModelBundle, days: int | None = None, output: str | Path | None = None,
              plan: pd.DataFrame | None = None, log: bool = True) -> pd.DataFrame:
    """plan: optional frame (store_id, sku_id, date, price, promo) of planned prices/promotions. Anything not
    in the plan uses the last known price and promo=0."""
    h = bundle.cfg.forecast.horizon
    days = min(days or h, h)
    last = last_sales_date(engine)
    if last is None:
        raise RuntimeError("sales_history is empty")
    recent = read_sales(engine, start=last - pd.Timedelta(days=LOOKBACK_DAYS))
    series = recent[["store_id", "sku_id"]].drop_duplicates()
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=days, freq="D")
    grid = series.merge(pd.DataFrame({"date": dates}), how="cross")
    if plan is not None:
        plan = plan.assign(date=pd.to_datetime(plan["date"]))
        grid = grid.merge(plan[["store_id", "sku_id", "date", "price", "promo"]], how="left", on=["store_id", "sku_id", "date"])
    instances = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in grid.to_dict("records")]

    out, rows = [], []
    for i in range(0, len(instances), CHUNK):
        res, lg = forecast(engine, bundle, instances[i : i + CHUNK], source="batch", request_id=f"batch-{last.date()}")
        out += res
        rows += lg
    if log:
        log_predictions(engine, rows)
    df = pd.DataFrame(out)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
    return df
