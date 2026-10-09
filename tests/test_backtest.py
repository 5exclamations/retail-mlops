import pandas as pd
import pytest

from retail_mlops.data.preprocess import preprocess
from retail_mlops.features.engineering import build_features, feature_columns, training_frame
from retail_mlops.models.backtest import fold_bounds, walk_forward
from retail_mlops.models.estimators import make_baselines, make_ridge

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def frame(raw, smoke_cfg):
    return training_frame(build_features(preprocess(raw)[0], smoke_cfg.forecast), smoke_cfg.forecast)


def test_folds_are_contiguous_and_end_at_test_start():
    ts = pd.Timestamp("2023-10-01")
    b = fold_bounds(ts, 3, 14)
    assert b[-1][1] == ts and all(b[i][1] == b[i + 1][0] for i in range(2)) and b[0][0] == ts - pd.Timedelta(days=42)


def test_walk_forward_never_reads_test_window(frame, smoke_cfg):
    f = smoke_cfg.forecast
    cols = feature_columns(f)
    ts = frame["date"].max() - pd.Timedelta(days=f.test_days - 1)
    pre = frame[frame["date"] < ts]
    poisoned = frame.copy()
    poisoned.loc[poisoned["date"] >= ts, "units"] = 1e6
    a = walk_forward(pre, cols, make_ridge(smoke_cfg), ts, 2, 14)
    b = walk_forward(poisoned[poisoned["date"] < ts], cols, make_ridge(smoke_cfg), ts, 2, 14)
    assert a["mean_wape"] == b["mean_wape"] and len(a["folds"]) == 2


def test_walk_forward_outputs(frame, smoke_cfg):
    ts = frame["date"].max() - pd.Timedelta(days=27)
    r = walk_forward(frame[frame["date"] < ts], feature_columns(smoke_cfg.forecast),
                     make_baselines(smoke_cfg)["seasonal_naive"], ts, 2, 14)
    assert set(r["by_lead"]) == set(range(1, 8)) and all(0 < f["wape"] < 2 for f in r["folds"])
    assert pd.Timestamp(r["folds"][0]["end"]) < pd.Timestamp(r["folds"][1]["start"])
