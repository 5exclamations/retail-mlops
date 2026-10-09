"""API behaviour: validation, security, maturity of ground truth, and Prometheus metrics."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from retail_mlops.config import GatesCfg
from retail_mlops.data.preprocess import preprocess
from retail_mlops.db import get_engine, init_db, read_sales, upsert_sales
from retail_mlops.models.train import run_training
from retail_mlops.pipelines.retrain import run_retraining
from retail_mlops.serving.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.api]
KEY = {"X-API-Key": "test-key"}


@pytest.fixture()
def client(cfg, raw, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-key")
    cfg.gates = GatesCfg(max_abs_bias=0.5)
    assert run_training(cfg, raw).outcome == "promoted"
    eng = get_engine(cfg.database_url)
    init_db(eng)
    upsert_sales(eng, preprocess(raw)[0])
    cfg.monitoring.refresh_seconds = cfg.monitoring.reload_check_seconds = 0
    cfg.monitoring.min_ground_truth = 1
    with TestClient(create_app(cfg)) as c:
        c.eng, c.cfg, c.last = eng, cfg, read_sales(eng).date.max()
        yield c


def inst(c, offset=3, skus=("K001",), stores=("S01", "S02", "S03")):
    d = (c.last + pd.Timedelta(days=offset)).date().isoformat()
    return [{"store_id": s, "sku_id": k, "date": d, "promo": 1} for s in stores for k in skus]


def test_predict_and_basic_endpoints(client):
    assert client.get("/health").status_code == 200 and client.get("/ready").status_code == 200
    r = client.post("/predict", json={"instances": inst(client)})
    assert r.status_code == 200
    ps = r.json()["predictions"]
    assert len(ps) == 3 and all(p["prediction"] >= 0 and p["horizon_days"] == 3 for p in ps)
    m = client.get("/model").json()
    assert m["version"] and m["backtest"]["mean_wape"] > 0 and m["model_sha256"]
    assert "text/html" in client.get("/dashboard").headers["content-type"]


@pytest.mark.parametrize("bad", [
    {"store_id": "../etc", "sku_id": "K001"}, {"store_id": "S01", "sku_id": "K001;DROP"},
    {"store_id": "S01", "sku_id": "K001", "price": -1}, {"store_id": "S01", "sku_id": "K001", "price": 1e9},
    {"store_id": "S01", "sku_id": "K001", "promo": 5}, {"store_id": "S01", "sku_id": "K001", "date": "1900-01-01"},
    {"store_id": "S01", "sku_id": "NOPE"}, {"store_id": "S01", "sku_id": "K001", "date": "2090-01-01"},
])
def test_bad_requests_are_rejected_with_4xx(client, bad):
    body = {"date": (client.last + pd.Timedelta(days=2)).date().isoformat(), **bad}
    assert client.post("/predict", json={"instances": [body]}).status_code == 422


def test_empty_and_oversized_batches_rejected(client):
    assert client.post("/predict", json={"instances": []}).status_code == 422
    assert client.post("/predict", json={"instances": inst(client) * 400}).status_code == 422


def test_admin_endpoints_require_key(client, monkeypatch):
    gt = {"records": [{"store_id": "S01", "sku_id": "K001", "date": "2023-01-02", "units": 1}]}
    assert client.post("/ground-truth", json=gt).status_code == 401
    assert client.post("/ground-truth", json=gt, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/admin/reload").status_code == 401
    assert client.post("/ground-truth", json=gt, headers=KEY).status_code == 200
    monkeypatch.delenv("ADMIN_API_KEY")  # unconfigured => disabled, not open
    assert client.post("/ground-truth", json=gt, headers=KEY).status_code == 403


def test_future_ground_truth_rejected(client):
    gt = {"records": [{"store_id": "S01", "sku_id": "K001", "date": "2099-01-01", "units": 1}]}
    assert client.post("/ground-truth", json=gt, headers=KEY).status_code == 422


def test_only_matured_true_forecasts_are_scored(client):
    # 1) retrospective: forecasting a day whose actual already exists must never count as accuracy evidence
    retro = [{"store_id": "S01", "sku_id": "K001", "date": client.last.date().isoformat()}] * 1
    client.post("/predict", json={"instances": retro})
    rep = client.post("/monitoring/refresh", headers=KEY).json()["performance"]
    assert rep["n_retrospective"] == 1 and rep["n_with_ground_truth"] == 0

    # 2) genuine forecasts stay pending until >= 95% of series have that day's actuals
    ins = inst(client, 2, skus=("K001", "K002"))
    client.post("/predict", json={"instances": ins})
    day = ins[0]["date"]
    part = [{"store_id": i["store_id"], "sku_id": i["sku_id"], "date": day, "units": 10.0} for i in ins[:2]]
    client.post("/ground-truth", json={"records": part}, headers=KEY)
    rep = client.post("/monitoring/refresh", headers=KEY).json()["performance"]
    assert rep["n_with_ground_truth"] == 0 and rep["n_pending"] >= len(ins)  # day not matured (partial actuals)

    # 3) once the day is complete for all series, forecasts are scored and broken down
    n_series = len(read_sales(client.eng)[["store_id", "sku_id"]].drop_duplicates())
    full = [{"store_id": f"S{s:02d}", "sku_id": f"K{k:03d}", "date": day, "units": 10.0}
            for s in range(1, 4) for k in range(1, 7)]
    assert len(full) == n_series
    client.post("/ground-truth", json={"records": full}, headers=KEY)
    rep = client.post("/monitoring/refresh", headers=KEY).json()["performance"]
    assert rep["status"] == "ok" and rep["n_with_ground_truth"] == len(ins)
    assert {"by_store", "by_sku", "by_category", "by_horizon"} <= set(rep) and list(rep["by_horizon"]) == ["2"]


def test_prometheus_metrics_after_traffic_and_retrain(client):
    for _ in range(3):
        client.post("/predict", json={"instances": inst(client)})
    client.post("/predict", json={"instances": [{"store_id": "S01", "sku_id": "NOPE", "date": "2023-10-28"}]})
    run_retraining(client.cfg, client.eng, force=True)  # no fresh data -> deferred, recorded in retrain_runs
    client.post("/monitoring/refresh", headers=KEY)
    text = client.get("/metrics").text
    assert 'retail_predictions_total{model_version="1",source="api"} 9.0' in text
    assert 'retail_prediction_errors_total{type="validation"} 1.0' in text
    assert 'retail_retrain_runs{status="deferred"} 1.0' in text
    for name in ("retail_model_inference_seconds_bucket", "retail_http_request_duration_seconds_bucket",
                 "retail_feature_psi", "retail_prediction_psi", "retail_model_info", "retail_model_trained_timestamp_seconds",
                 "retail_monitoring_last_refresh_timestamp_seconds", "retail_pending_predictions", "retail_prediction_units_bucket"):
        assert name in text, name
    assert client.get("/retraining/history").json()[0]["status"] == "deferred"
