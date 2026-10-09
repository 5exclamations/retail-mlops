import numpy as np
import pandas as pd
import pytest

from retail_mlops.data.generate import DriftSpec, generate_sales
from retail_mlops.data.preprocess import preprocess
from retail_mlops.data.validate import validate_raw

pytestmark = pytest.mark.unit


def test_generation_is_deterministic(smoke_cfg):
    pd.testing.assert_frame_equal(generate_sales(smoke_cfg, 60), generate_sales(smoke_cfg, 60))


def test_generated_data_passes_validation(raw, smoke_cfg):
    assert validate_raw(raw, smoke_cfg.validation).passed


def test_weekend_effect_present(raw):
    d = raw.drop_duplicates(["store_id", "sku_id", "date"])
    by_dow = d.groupby(pd.to_datetime(d["date"]).dt.dayofweek)["units"].mean()
    assert by_dow[5] > by_dow[0]


def test_drift_scenario_changes_demand(smoke_cfg):
    base = generate_sales(smoke_cfg, 120)
    drifted = generate_sales(smoke_cfg, 120, DriftSpec("2023-03-01", {"beverages": 2.0}, 1.0, 1.0))
    m = lambda df: df[(df.category == "beverages") & (pd.to_datetime(df.date) >= "2023-03-01")].units.mean()  # noqa: E731
    assert m(drifted) > 1.5 * m(base)


@pytest.mark.parametrize("mutate,check", [
    (lambda d: d.drop(columns=["price"]), "schema"),
    (lambda d: d.assign(units=d["units"].where(d.index % 5 != 0, -3)), "units_non_negative"),
    (lambda d: d.assign(price=d["price"].where(d.index % 4 != 0, np.nan)), "nulls_price"),
    (lambda d: pd.concat([d, d.head(len(d) // 4)]), "duplicate_keys"),
    (lambda d: d.assign(promo=2), "promo_binary"),
    (lambda d: d.drop(d.sample(frac=0.2, random_state=1).index), "panel_completeness"),
])
def test_validation_catches_bad_data(raw, smoke_cfg, mutate, check):
    rep = validate_raw(mutate(raw.copy()), smoke_cfg.validation)
    assert not rep.passed
    assert check in {c.name for c in rep.errors}


def test_preprocess_cleans(raw):
    clean, rep = preprocess(raw)
    assert not clean.duplicated(["store_id", "sku_id", "date"]).any()
    assert clean["price"].notna().all() and set(clean["promo"]) <= {0, 1}
    assert rep.duplicates_dropped > 0 and rep.prices_imputed > 0
    assert (clean.groupby(["store_id", "sku_id"])["date"].diff().dropna() == pd.Timedelta(days=1)).all()
