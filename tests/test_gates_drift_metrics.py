import numpy as np
import pandas as pd
import pytest

from retail_mlops.config import GatesCfg, MonitoringCfg
from retail_mlops.models.metrics import compute_metrics
from retail_mlops.monitoring.drift import drift_report, psi_categorical, psi_numeric
from retail_mlops.registry.gates import evaluate_gates, paired_bootstrap

pytestmark = pytest.mark.unit
M = lambda wape, bias=0.0: {"wape": wape, "bias": bias}  # noqa: E731


def gate(cand, base=M(0.4), champ=None, seg=None, cfg=None, paired=None, holdout=30):
    seg = seg or {"a": M(0.3)}
    return evaluate_gates(cfg or GatesCfg(), cand, base, seg, {"a": M(0.4)}, champ, True, paired, holdout)


SIG = {"rel_improvement": 0.05, "ci_low": 0.02, "ci_high": 0.08, "n_days": 30}


def test_metrics():
    m = compute_metrics([10, 10], [12, 6])
    assert m["mae"] == 3 and m["wape"] == pytest.approx(0.3) and m["bias"] == pytest.approx(-0.1)


def test_first_model_needs_only_basic_gates():
    assert gate(M(0.3)).passed


@pytest.mark.parametrize("kw,failed", [
    (dict(cand=M(0.39)), "beats_baseline"),
    (dict(cand=M(0.3, 0.2)), "bias_within_bound"),
    (dict(cand=M(0.3), seg={"a": M(0.5)}), "no_segment_regression"),
])
def test_basic_gate_failures(kw, failed):
    res = gate(**kw)
    assert not res.passed and failed in [c.name for c in res.checks if not c.passed]
    assert any(r.startswith(failed) for r in res.reasons)  # clear rejection reason


def test_clear_win_over_champion_promotes():
    assert gate(M(0.30), champ=M(0.315), paired=SIG).passed


def test_tie_with_champion_is_rejected():
    res = gate(M(0.3016), champ=M(0.3017), paired={"rel_improvement": 0.0003, "ci_low": -0.01, "ci_high": 0.01, "n_days": 30})
    failed = {c.name for c in res.checks if not c.passed}
    assert not res.passed and {"improves_on_champion", "significant_vs_champion"} <= failed


def test_improvement_above_threshold_but_insignificant_is_rejected():
    res = gate(M(0.30), champ=M(0.315), paired={"rel_improvement": 0.05, "ci_low": -0.02, "ci_high": 0.1, "n_days": 30})
    assert [c.name for c in res.checks if not c.passed] == ["significant_vs_champion"]


def test_significance_requirement_is_configurable():
    cfg = GatesCfg(require_significance=False)
    assert gate(M(0.30), champ=M(0.315), cfg=cfg, paired=None).passed


def test_min_improvement_is_configurable():
    assert not gate(M(0.30), champ=M(0.315), paired=SIG, cfg=GatesCfg(min_improvement_vs_champion=0.10)).passed


def test_insufficient_fresh_holdout_blocks_promotion():
    res = gate(M(0.30), champ=M(0.315), paired=SIG, holdout=3)
    assert not res.passed and "fresh_holdout" in [c.name for c in res.checks if not c.passed]


def test_paired_bootstrap_detects_real_gain_and_ignores_noise():
    rng = np.random.default_rng(0)
    n_days, n_series = 40, 30
    d = pd.DataFrame({"date": np.repeat(np.arange(n_days), n_series), "y": rng.poisson(20, n_days * n_series).astype(float)})
    d["p_champion"] = d["y"] + rng.normal(0, 6, len(d))
    same = d.assign(p_candidate=d["p_champion"])
    assert paired_bootstrap(same, 500, 0.05, 0)["rel_improvement"] == 0
    better = d.assign(p_candidate=d["y"] + rng.normal(0, 3, len(d)))
    assert paired_bootstrap(better, 500, 0.05, 0)["ci_low"] > 0
    noise = d.assign(p_candidate=d["y"] + rng.normal(0, 6, len(d)))
    assert paired_bootstrap(noise, 500, 0.05, 0)["ci_low"] < 0.02


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
