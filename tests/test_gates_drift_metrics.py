import numpy as np
import pandas as pd
import pytest

from retail_mlops.config import GatesCfg, MonitoringCfg
from retail_mlops.models.metrics import compute_metrics
from retail_mlops.monitoring.drift import drift_report, psi_categorical, psi_numeric
from retail_mlops.registry.gates import evaluate_gates

pytestmark = pytest.mark.unit
M = lambda wape, bias=0.0: {"wape": wape, "bias": bias}  # noqa: E731


def test_metrics():
    m = compute_metrics([10, 10], [12, 6])
    assert m["mae"] == 3 and m["wape"] == pytest.approx(0.3) and m["bias"] == pytest.approx(-0.1)


def gate(cand, base=M(0.4), champ=None, seg=None, cfg=None):
    seg = seg or {"a": M(0.3)}
    return evaluate_gates(cfg or GatesCfg(), cand, base, seg, {"a": M(0.4)}, champ, True)


def test_gate_passes_good_candidate():
    assert gate(M(0.3)).passed


@pytest.mark.parametrize("kw,failed", [
    (dict(cand=M(0.39)), "beats_baseline"),
    (dict(cand=M(0.3, 0.2)), "bias_within_bound"),
    (dict(cand=M(0.3), seg={"a": M(0.5)}), "no_segment_regression"),
    (dict(cand=M(0.3), champ=M(0.28)), "not_worse_than_champion"),
])
def test_gate_failures(kw, failed):
    res = gate(**kw)
    assert not res.passed and failed in [c.name for c in res.checks if not c.passed]


def test_psi_identity_and_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(a, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert psi_numeric(a, pd.Series(rng.normal(1.5, 1, 5000))) > 0.5
    assert psi_categorical(pd.Series(list("aabb") * 100), pd.Series(list("aaab") * 100)) > 0.05


def test_drift_report_flags_shifted_feature_only():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 4000), "y": rng.normal(0, 1, 4000)})
    cur = pd.DataFrame({"x": rng.normal(3, 1, 1000), "y": rng.normal(0, 1, 1000)})
    r = drift_report(ref, cur, ["x", "y"], MonitoringCfg(min_samples=100, drift_share_alert=0.5))
    st = {f["feature"]: f["status"] for f in r["features"]}
    assert st == {"x": "alert", "y": "ok"} and r["drifted"]


def test_drift_insufficient_data():
    r = drift_report(pd.DataFrame({"x": [1.0] * 10}), pd.DataFrame({"x": [1.0] * 5}), ["x"], MonitoringCfg(min_samples=50))
    assert r["status"] == "insufficient_data"
