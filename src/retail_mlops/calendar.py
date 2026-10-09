"""Retail calendar shared by the data generator and the feature pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def holidays_for_year(year: int) -> dict[pd.Timestamp, str]:
    thanksgiving = _nth_weekday(year, 11, 3, 4)  # 4th Thursday of November
    return {
        pd.Timestamp(year, 1, 1): "new_year",
        pd.Timestamp(year, 2, 14): "valentines",
        pd.Timestamp(year, 7, 4): "independence_day",
        thanksgiving: "thanksgiving",
        thanksgiving + pd.Timedelta(days=1): "black_friday",
        pd.Timestamp(year, 12, 24): "christmas_eve",
        pd.Timestamp(year, 12, 25): "christmas",
    }


def holiday_frame(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """is_holiday, is_black_friday, days_to_holiday (capped at 14; 14 means 'none soon')."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    years = range(idx.year.min() - 1, idx.year.max() + 2)
    hol = sorted(d for y in years for d in holidays_for_year(y))
    hol_arr = np.array(hol, dtype="datetime64[ns]")
    day = idx.values.astype("datetime64[ns]")
    pos = np.searchsorted(hol_arr, day, side="left")
    nxt = hol_arr[np.clip(pos, 0, len(hol_arr) - 1)]
    days_to = ((nxt - day) / np.timedelta64(1, "D")).astype(int)
    black_friday = {d for y in years for d, n in holidays_for_year(y).items() if n == "black_friday"}
    return pd.DataFrame(
        {
            "is_holiday": (days_to == 0).astype(int),
            "is_black_friday": idx.isin(list(black_friday)).astype(int),
            "days_to_holiday": np.minimum(days_to, 14),
        }
    )
