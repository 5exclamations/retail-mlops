"""Automated cleaning: raw POS-style export -> complete, typed store x SKU x day panel."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from retail_mlops.data.validate import KEY

SERIES = ["store_id", "sku_id"]


@dataclass
class PreprocessReport:
    rows_in: int = 0
    rows_out: int = 0
    duplicates_dropped: int = 0
    prices_imputed: int = 0
    promo_imputed: int = 0
    days_filled: int = 0
    units_clipped: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "notes"}


def complete_panel(df: pd.DataFrame, end_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Reindex each series to a gap-free daily calendar (new rows have NaN units/price/promo)."""
    out = []
    for (store, sku), g in df.groupby(SERIES, sort=False):
        last = max(g["date"].max(), end_date) if end_date is not None else g["date"].max()
        idx = pd.date_range(g["date"].min(), last, freq="D", name="date")
        g = g.set_index("date").reindex(idx)
        g["store_id"], g["sku_id"], g["category"] = store, sku, g["category"].dropna().iloc[0]
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True) if out else df


def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessReport]:
    rep = PreprocessReport(rows_in=len(raw))
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ["units", "price", "promo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n = len(df)
    df = df.drop_duplicates(KEY, keep="last")  # later export wins
    rep.duplicates_dropped = n - len(df)

    df = df.sort_values(KEY)
    before = len(df)
    df = complete_panel(df)
    rep.days_filled = len(df) - before

    rep.prices_imputed = int(df["price"].isna().sum())
    df["price"] = df.groupby(SERIES)["price"].transform(lambda s: s.ffill().bfill())
    rep.promo_imputed = int(df["promo"].isna().sum())
    df["promo"] = df["promo"].fillna(0).astype(int)  # unknown promo status treated as "no promo"

    neg = df["units"] < 0
    rep.units_clipped = int(neg.sum())
    df.loc[neg, "units"] = 0.0  # negative units are returns, not demand

    df = df.sort_values(KEY).reset_index(drop=True)
    rep.rows_out = len(df)
    return df, rep
