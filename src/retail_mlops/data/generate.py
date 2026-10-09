"""Synthetic retail sales generator.

Demand for each store x SKU x day is a count drawn from a gamma-Poisson (negative binomial) whose mean is the
product of: base level, store size, linear trend, weekday profile, yearly seasonality, holiday uplift,
promotion lift and a price-elasticity term. A small fraction of days are stock-outs (units forced to 0).
The generator is fully seeded and optionally applies a *drift scenario* from a given date, which the demo uses
to exercise monitoring and retraining.

Nothing here is real data; it only needs to be structured enough that models can be compared honestly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from retail_mlops.calendar import holiday_frame
from retail_mlops.config import Config

# weekday profile Mon..Sun, yearly amplitude / peak day-of-year, price elasticity, promo lift, holiday uplift
CATEGORIES: dict[str, dict] = {
    "beverages": dict(base=(18, 45), price=(1.5, 4.0), week=[0.9, 0.9, 0.95, 1.0, 1.15, 1.3, 1.2],
                      amp=0.35, peak=200, elast=-1.6, promo=0.45, hol=1.25),
    "snacks": dict(base=(12, 35), price=(2.0, 6.0), week=[0.85, 0.9, 0.95, 1.0, 1.2, 1.35, 1.15],
                   amp=0.10, peak=330, elast=-1.3, promo=0.55, hol=1.30),
    "produce": dict(base=(25, 70), price=(1.0, 5.0), week=[0.95, 1.0, 1.0, 1.0, 1.1, 1.25, 1.0],
                    amp=0.20, peak=190, elast=-0.9, promo=0.30, hol=1.15),
    "household": dict(base=(6, 20), price=(4.0, 15.0), week=[1.05, 1.0, 1.0, 1.0, 1.0, 1.1, 0.85],
                      amp=0.05, peak=80, elast=-1.1, promo=0.40, hol=0.95),
    "seasonal": dict(base=(5, 18), price=(8.0, 30.0), week=[0.8, 0.85, 0.9, 1.0, 1.2, 1.4, 1.3],
                     amp=0.45, peak=350, elast=-2.0, promo=0.60, hol=1.9),
}


@dataclass
class DriftSpec:
    """Distribution shift applied to all dates >= start_date."""
    start_date: str
    category_demand_multiplier: dict[str, float] = field(default_factory=lambda: {"beverages": 1.35, "snacks": 0.8})
    promo_lift_scale: float = 0.5
    price_multiplier: float = 1.10


def _series_table(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng([cfg.seed, 1])
    cats = list(CATEGORIES)
    store_scale = rng.uniform(0.6, 1.6, cfg.data.n_stores)
    rows = []
    for k in range(cfg.data.n_skus):
        cat = cats[k % len(cats)]
        spec = CATEGORIES[cat]
        base = rng.uniform(*spec["base"])
        price = round(float(rng.uniform(*spec["price"])), 2)
        trend = rng.normal(0.05, 0.08)
        for s in range(cfg.data.n_stores):
            rows.append(dict(store_id=f"S{s + 1:02d}", sku_id=f"K{k + 1:03d}", category=cat,
                             base=base * store_scale[s] * rng.uniform(0.9, 1.1), base_price=price, trend=trend))
    return pd.DataFrame(rows)


def _promo_flags(rng: np.random.Generator, n_days: int, start_prob: float = 0.025) -> np.ndarray:
    flags = np.zeros(n_days, dtype=int)
    starts = np.flatnonzero(rng.random(n_days) < start_prob)
    for st in starts:
        flags[st : st + int(rng.integers(3, 8))] = 1
    return flags


def generate_sales(cfg: Config, n_days: int | None = None, drift: DriftSpec | None = None) -> pd.DataFrame:
    n_days = n_days or cfg.data.n_days
    dates = pd.date_range(cfg.data.start_date, periods=n_days, freq="D")
    hol = holiday_frame(dates)
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    t_years = np.arange(n_days) / 365.0
    drift_mask = np.asarray(dates >= pd.Timestamp(drift.start_date)) if drift else np.zeros(n_days, bool)

    frames = []
    for i, row in enumerate(_series_table(cfg).itertuples(index=False)):
        spec = CATEGORIES[row.category]
        rng = np.random.default_rng([cfg.seed, 2, i])
        promo = _promo_flags(rng, n_days)
        discount = np.where(promo == 1, rng.uniform(0.10, 0.30, n_days), 0.0)
        price = np.round(row.base_price * (1 - discount) * (1 + rng.normal(0, 0.01, n_days)), 2)
        promo_lift = spec["promo"] * np.where(drift_mask, drift.promo_lift_scale, 1.0) if drift else spec["promo"]
        level_mult = np.ones(n_days)
        if drift:
            price = np.where(drift_mask, np.round(price * drift.price_multiplier, 2), price)
            level_mult = np.where(drift_mask, drift.category_demand_multiplier.get(row.category, 1.0), 1.0)
        # elasticity is relative to the SKU's reference price so general inflation also moves demand
        price_effect = (price / row.base_price) ** spec["elast"]
        mean = (
            row.base * (1 + row.trend * t_years)
            * np.asarray(spec["week"])[dow]
            * (1 + spec["amp"] * np.cos(2 * np.pi * (doy - spec["peak"]) / 365.0))
            * np.where(hol["is_holiday"].to_numpy() == 1, spec["hol"], 1.0)
            * (1 + promo_lift * promo)
            * price_effect
            * level_mult
        )
        k = 10.0  # dispersion: var = mean + mean^2 / k
        lam = mean * rng.gamma(k, 1.0 / k, n_days)
        units = rng.poisson(lam).astype(float)
        units[rng.random(n_days) < 0.004] = 0.0  # stock-outs (censored demand)
        frames.append(pd.DataFrame(dict(
            date=dates, store_id=row.store_id, sku_id=row.sku_id, category=row.category,
            units=units, price=price, promo=promo,
        )))
    df = pd.concat(frames, ignore_index=True)
    if cfg.data.inject_quality_issues:
        df = _inject_quality_issues(df, np.random.default_rng([cfg.seed, 3]))
    return df.sort_values(["store_id", "sku_id", "date"], kind="stable").reset_index(drop=True)


def _inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mimic a messy POS export: missing prices/promo flags and a few duplicated rows."""
    df = df.copy()
    df.loc[rng.random(len(df)) < 0.0015, "price"] = np.nan
    df["promo"] = df["promo"].astype(float)
    df.loc[rng.random(len(df)) < 0.0005, "promo"] = np.nan
    dups = df.sample(frac=0.001, random_state=int(rng.integers(0, 2**31)))
    return pd.concat([df, dups], ignore_index=True)
