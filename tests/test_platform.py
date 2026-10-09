"""Integration: train -> register -> serve -> monitor -> retrain, on the small smoke dataset."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from retail_mlops.config import GatesCfg
from retail_mlops.data.preprocess import preprocess
from retail_mlops.db import get_engine, init_db, read_sales, upsert_sales
from retail_mlops.models.train import run_training
from retail_mlops.pipelines.batch_predict import run_batch
from retail_mlops.pipelines.retrain import decide, run_retraining
from retail_mlops.registry import store
from retail_mlops.registry.bundle import load_bundle
from retail_mlops.serving.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.api]


@pytest.fixture()
def trained(cfg, raw):
    cfg.gates = GatesCfg(max_abs_bias=0.5)  # smoke data is tiny; bias is exercised in test_gates
    res = run_training(cfg, raw)
    assert res.outcome == "promoted", res.gates.summary()
    eng = get_engine(cfg.database_url)
    init_db(eng)
    upsert_sales(eng, preprocess(raw)[0])
    return cfg, eng, res


def test_train_registers_and_sets_champion(trained):
    cfg, _, res = trained
    c = store.client_for(cfg)
    assert store.alias_version(c, cfg, "champion") == res.version
    b = load_bundle(cfg)
    assert b.reference is not None and "__prediction" in b.reference and b.metadata["data_hash"]


def test_rejected_candidate_not_promoted(trained):
    cfg, _, res = trained
    from retail_mlops.data.generate import generate_sales
    cfg.gates = GatesCfg(min_improvement_vs_baseline=0.99)  # impossible bar
    res2 = run_training(cfg, generate_sales(cfg))
    assert res2.outcome == "rejected"
    assert store.alias_version(store.client_for(cfg), cfg, "champion") == res.version


def test_invalid_data_blocks_training(cfg, raw):
    res = run_training(cfg, raw.assign(units=-1.0))
    assert res.outcome == "data_invalid" and res.version is None


def test_rollback(trained):
    cfg, _, res = trained
    cfg.gates = GatesCfg(max_abs_bias=0.5, max_regression_vs_champion=1.0)
    from retail_mlops.data.generate import generate_sales
    res2 = run_training(cfg, generate_sales(cfg))
    assert res2.outcome == "promoted"
    assert store.rollback(store.client_for(cfg), cfg) == res.version


def test_batch_predict(trained, tmp_path):
    cfg, eng, _ = trained
    df = run_batch(eng, load_bundle(cfg), output=tmp_path / "f.csv")
    assert len(df) == 3 * 6 * 7 and (df.prediction >= 0).all() and (tmp_path / "f.csv").exists()


def test_api_end_to_end(trained):
    cfg, eng, _ = trained
    cfg.monitoring.refresh_seconds = cfg.monitoring.reload_check_seconds = 0
    last = read_sales(eng).date.max()
    with TestClient(create_app(cfg)) as c:
        assert c.get("/health").status_code == 200 and c.get("/ready").status_code == 200
        d = (last + pd.Timedelta(days=3)).date().isoformat()
        inst = [{"store_id": s, "sku_id": "K001", "date": d, "promo": 1} for s in ("S01", "S02", "S03")]
        for _ in range(70):  # enough logged predictions to clear min_samples
            r = c.post("/predict", json={"instances": inst})
        body = r.json()
        assert r.status_code == 200 and len(body["predictions"]) == 3 and all(p["prediction"] >= 0 for p in body["predictions"])
        # error paths
        assert c.post("/predict", json={"instances": []}).status_code == 422
        far = c.post("/predict", json={"instances": [{**inst[0], "date": (last + pd.Timedelta(days=30)).date().isoformat()}]})
        assert far.status_code == 422 and "max forecast lead" in str(far.json())
        assert c.post("/predict", json={"instances": [{**inst[0], "sku_id": "NOPE"}]}).status_code == 422
        # ground truth arrives -> live performance
        gt = [{"store_id": s, "sku_id": "K001", "date": d, "units": 20.0} for s in ("S01", "S02", "S03")]
        g = c.post("/ground-truth", json={"records": gt})
        assert g.status_code == 200, g.text
        assert g.json()["upserted"] == 3
        cfg.monitoring.min_ground_truth = 1
        rep = c.post("/monitoring/refresh").json()
        assert rep["performance"]["n_with_ground_truth"] >= 3 and rep["drift"]["status"] == "ok"
        assert rep["prediction_drift"]["psi"] is not None
        m = c.get("/metrics").text
        for name in ("retail_http_requests_total", "retail_model_inference_seconds_bucket", "retail_feature_psi",
                     "retail_live_wape", "retail_prediction_errors_total", "retail_model_info"):
            assert name in m, name
        assert c.get("/model").json()["version"]


def test_retrain_decision_triggers(trained):
    cfg, eng, _ = trained
    b = load_bundle(cfg)
    assert decide(cfg, eng, b)[0] == []
    assert decide(cfg, eng, b, force=True)[0] == ["forced"]
    assert decide(cfg, eng, None)[0][0].startswith("bootstrap")
    cfg.retrain.max_model_age_days = 0
    assert any(r.startswith("schedule") for r in decide(cfg, eng, b)[0])


def test_retrain_pipeline_runs(trained):
    cfg, eng, _ = trained
    cfg.gates = GatesCfg(max_abs_bias=0.5, max_regression_vs_champion=1.0)
    out = run_retraining(cfg, eng, force=True)
    assert out["status"] in ("promoted", "rejected") and out["mlflow_run_id"]
    assert run_retraining(cfg, eng)["status"] == "skipped"
