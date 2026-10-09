"""Raw-data validation. Returns a structured report; errors block training, warnings do not."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from retail_mlops.config import ValidationCfg

REQUIRED_COLUMNS = ["date", "store_id", "sku_id", "category", "units", "price", "promo"]
KEY = ["store_id", "sku_id", "date"]


@dataclass
class Check:
    name: str
    passed: bool
    severity: str  # "error" | "warning"
    detail: str


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    n_rows: int = 0

    def add(self, name: str, passed: bool, detail: str, severity: str = "error") -> None:
        self.checks.append(Check(name, bool(passed), severity, detail))

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "n_rows": self.n_rows, "checks": [asdict(c) for c in self.checks]}

    def summary(self) -> str:
        lines = [f"data validation: {'PASSED' if self.passed else 'FAILED'} ({self.n_rows} rows)"]
        lines += [f"  [{'ok' if c.passed else c.severity.upper()}] {c.name}: {c.detail}" for c in self.checks]
        return "\n".join(lines)


def validate_raw(df: pd.DataFrame, cfg: ValidationCfg) -> ValidationReport:
    rep = ValidationReport(n_rows=len(df))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    rep.add("schema", not missing, f"missing columns: {missing}" if missing else "all required columns present")
    if missing or df.empty:
        if df.empty:
            rep.add("non_empty", False, "dataframe is empty")
        return rep

    dates = pd.to_datetime(df["date"], errors="coerce")
    rep.add("date_parse", dates.notna().all(), f"{int(dates.isna().sum())} unparseable dates")

    for col in ["units", "price", "promo"]:
        frac = df[col].isna().mean()
        rep.add(f"nulls_{col}", frac <= cfg.max_null_fraction, f"{frac:.4%} null (max {cfg.max_null_fraction:.2%})")
    for col in ["store_id", "sku_id", "category"]:
        n = int(df[col].isna().sum())
        rep.add(f"nulls_{col}", n == 0, f"{n} nulls")

    units = pd.to_numeric(df["units"], errors="coerce")
    rep.add("units_non_negative", (units.dropna() >= 0).all(), f"{int((units < 0).sum())} negative values")
    rep.add("units_upper_bound", (units.dropna() <= cfg.max_units).all(), f"max={units.max()}")
    price = pd.to_numeric(df["price"], errors="coerce")
    rep.add("price_positive", (price.dropna() > 0).all(), f"{int((price <= 0).sum())} non-positive prices")
    promo = df["promo"].dropna()
    rep.add("promo_binary", promo.isin([0, 1]).all(), f"{int((~promo.isin([0, 1])).sum())} non-binary flags")

    dup = df.duplicated(KEY).mean()
    rep.add("duplicate_keys", dup <= cfg.max_duplicate_fraction, f"{dup:.4%} duplicated keys (max {cfg.max_duplicate_fraction:.2%})")
    cat_per_sku = df.groupby("sku_id")["category"].nunique()
    rep.add("category_consistent", (cat_per_sku == 1).all(), f"{int((cat_per_sku > 1).sum())} SKUs with >1 category")

    # panel completeness: every store x sku should have one row per day between its first and last date
    d = df.assign(date=dates).dropna(subset=["date"]).drop_duplicates(KEY)
    span = d.groupby(["store_id", "sku_id"])["date"].agg(["min", "max", "count"])
    expected = (span["max"] - span["min"]).dt.days + 1
    missing_frac = float(1 - span["count"].sum() / expected.sum())
    rep.add("panel_completeness", missing_frac <= cfg.max_missing_day_fraction,
            f"{missing_frac:.4%} of expected store-sku-days missing (max {cfg.max_missing_day_fraction:.2%})")
    short = span[(span["max"] - span["min"]).dt.days + 1 < cfg.min_history_days]
    rep.add("min_history", short.empty, f"{len(short)} series shorter than {cfg.min_history_days} days")

    # robust outlier screen: a warning only, because genuine spikes happen
    med = d.groupby(["store_id", "sku_id"])["units"].transform("median").clip(lower=1)
    n_out = int((d["units"] > cfg.outlier_multiple_warn * med).sum())
    rep.add("outliers", n_out == 0, f"{n_out} rows above {cfg.outlier_multiple_warn}x series median", severity="warning")
    return rep
