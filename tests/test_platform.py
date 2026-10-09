"""Integration: train -> register -> compare on fresh holdout -> promote/reject -> rollback -> serve -> monitor.

The synthetic history is split in two: the first CUT days exist when v1 trains; the rest arrives later and is the
fresh data on which any challenger and the champion are compared."""
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from retail_mlops.config import GatesCfg
from retail_mlops.data.preprocess import preprocess
from retail_mlops.db import get_engine, init_db, upsert_sales
from retail_mlops.models.train import run_training
from retail_mlops.pipelines.batch_predict import run_batch
from retail_mlops.pipelines.retrain import decide, run_retraining
from retail_mlops.registry import store
from retail_mlops.registry.bundle import load_bundle
from retail_mlops.registry.integrity import IntegrityError

pytestmark = [pytest.mark.integration, pytest.mark.api]
CUT = pd.Timestamp("2023-07-19")  # 200th day of the smoke dataset


def early(raw):
    return raw[pd.to_datetime(raw["date"]) <= CUT]


@pytest.fixture()
def world(cfg, raw):
    """v1 promoted on early data; DB holds early data only. Later data is delivered by `deliver()`."""
    cfg.gates = GatesCfg(max_abs_bias=0.5, min_holdout_days=7)
    res = run_training(cfg, early(raw))
    assert res.outcome == "promoted", res.reasons
    eng = get_engine(cfg.database_url)
    init_db(eng)
    clean = preprocess(raw)[0]
    upsert_sales(eng, clean[clean["date"] <= CUT])

    def deliver():
        upsert_sales(eng, clean[clean["date"] > CUT])

    return cfg, eng, res, deliver


def champion(cfg):
    return store.alias_version(store.client_for(cfg), cfg, "champion")


def test_first_model_registered_with_provenance(world):
    cfg, _, res, _ = world
    mv = store.client_for(cfg).get_model_version(cfg.mlflow.model_name, res.version)
    for tag in ("model_sha256", "data_hash", "data_end", "test_start", "git_sha", "test_wape"):
        assert mv.tags.get(tag)
    b = load_bundle(cfg)
    assert b.metadata["train_end"] < b.metadata["test_start"] <= b.metadata["data_end"]
    assert "backtest" in b.metadata["metrics"] and "__prediction" in b.reference


def test_challenger_is_compared_only_on_data_after_champion_data_end(world, raw):
    cfg, _, v1, _ = world
    cfg.gates = GatesCfg(max_abs_bias=0.5, min_holdout_days=7, require_significance=False,
                         min_improvement_vs_champion=-1.0)
    res = run_training(cfg, raw)
    assert res.outcome == "promoted", res.reasons
    v2 = load_bundle(cfg, version=res.version).metadata
    v1_meta = load_bundle(cfg, version=v1.version).metadata
    assert pd.Timestamp(v2["test_start"]) > pd.Timestamp(v1_meta["data_end"])  # champion never saw the holdout
    assert pd.Timestamp(v2["train_end"]) < pd.Timestamp(v2["test_start"])  # challenger never trained on it
    assert v2["champion_version_compared"] == v1.version and v2["champion_test"]["n"] == v2["metrics"]["test"]["n"]
    assert v2["holdout_days"] >= 7


def test_no_fresh_data_means_no_comparison_and_no_new_version(world, raw):
    cfg, _, v1, _ = world
    res = run_training(cfg, early(raw))  # same data the champion already saw
    assert res.outcome == "insufficient_holdout" and "fair comparison" in res.reasons[0]
    assert champion(cfg) == v1.version
    assert len(store.client_for(cfg).search_model_versions(f"name='{cfg.mlflow.model_name}'")) == 1


def test_rejection_keeps_champion_and_explains_why(world, raw):
    cfg, _, v1, _ = world
    cfg.gates = GatesCfg(max_abs_bias=0.5, min_holdout_days=7, min_improvement_vs_champion=0.9)
    res = run_training(cfg, raw)
    assert res.outcome == "rejected" and any(r.startswith("improves_on_champion") for r in res.reasons)
    assert champion(cfg) == v1.version
    mv = store.client_for(cfg).get_model_version(cfg.mlflow.model_name, res.version)
    assert mv.tags["stage"] == "rejected" and "improves_on_champion" in mv.tags["rejection_reasons"]


def test_promote_then_rollback_preserves_all_versions(world, raw):
    cfg, _, v1, _ = world
    cfg.gates = GatesCfg(max_abs_bias=0.5, min_holdout_days=7, require_significance=False,
                         min_improvement_vs_champion=-1.0)
    v2 = run_training(cfg, raw)
    assert champion(cfg) == v2.version
    client = store.client_for(cfg)
    assert store.alias_version(client, cfg, "previous") == v1.version
    assert store.rollback(client, cfg) == v1.version and champion(cfg) == v1.version
    for v in (v1.version, v2.version):  # historical versions still load, with metadata intact
        b = load_bundle(cfg, version=v)
        assert b.metadata["data_hash"] and b.metadata["model_sha256"]


def test_training_is_reproducible(cfg, raw, tmp_path):
    outs = []
    for i in range(2):
        c = cfg.model_copy(deep=True)
        c.mlflow.tracking_uri = f"sqlite:///{tmp_path}/repro{i}.db"
        c.gates = GatesCfg(max_abs_bias=0.5)
        r = run_training(c, raw)
        b = load_bundle(c)
        outs.append((r.metrics["test"], b.metadata["data_hash"], b.metadata["model_sha256"], b))
    assert outs[0][0] == outs[1][0] and outs[0][1] == outs[1][1]
    ref = outs[0][3].reference.drop(columns="__prediction").head(200)
    assert np.allclose(outs[0][3].predict_features(ref), outs[1][3].predict_features(ref))
    # note: the .skops zip is not byte-identical across runs (archive metadata), so integrity hashes differ;
    # reproducibility is asserted on metrics, data hash and predictions.
    assert outs[0][2] != "" and outs[1][2] != ""


def test_tampered_registry_hash_blocks_loading(world):
    cfg, _, res, _ = world
    store.client_for(cfg).set_model_version_tag(cfg.mlflow.model_name, res.version, "model_sha256", "0" * 64)
    with pytest.raises(IntegrityError):
        load_bundle(cfg)


def test_missing_hash_blocks_loading(world):
    cfg, _, res, _ = world
    store.client_for(cfg).delete_model_version_tag(cfg.mlflow.model_name, res.version, "model_sha256")
    with pytest.raises(IntegrityError):
        load_bundle(cfg)


def test_invalid_data_blocks_training(cfg, raw):
    res = run_training(cfg, raw.assign(units=-1.0))
    assert res.outcome == "data_invalid" and res.version is None


def test_batch_predict_records_horizons(world, tmp_path):
    cfg, eng, _, _ = world
    df = run_batch(eng, load_bundle(cfg), output=tmp_path / "f.csv")
    assert len(df) == 3 * 6 * 7 and (df.prediction >= 0).all()
    assert sorted(df.horizon_days.unique()) == list(range(1, 8))


def test_retrain_decision_triggers(world):
    cfg, eng, _, deliver = world
    b = load_bundle(cfg)
    assert decide(cfg, eng, b)[0] == []
    assert decide(cfg, eng, b, force=True)[0] == ["forced"]
    assert decide(cfg, eng, None)[0][0].startswith("bootstrap")
    cfg.retrain.max_model_age_days = 0
    assert any(r.startswith("schedule") for r in decide(cfg, eng, b)[0])
    cfg.retrain.max_model_age_days, cfg.retrain.min_new_rows = 999, 100
    deliver()
    assert any(r.startswith("data") for r in decide(cfg, eng, b)[0])


def test_retrain_pipeline_defers_without_fresh_data_then_runs_with_it(world):
    cfg, eng, _, deliver = world
    out = run_retraining(cfg, eng, force=True)
    assert out["status"] == "deferred" and out["rejection_reasons"]
    deliver()
    cfg.gates = GatesCfg(max_abs_bias=0.5, min_holdout_days=7, min_improvement_vs_champion=0.9)
    out = run_retraining(cfg, eng, force=True)
    assert out["status"] == "rejected" and out["model_version"] and out["rejection_reasons"]
    assert run_retraining(cfg, eng)["status"] == "skipped"


def test_history_is_listable(world, capsys):
    from retail_mlops.cli import main

    cfg, _, _, _ = world
    import os
    os.environ["MLFLOW_TRACKING_URI"] = cfg.mlflow.tracking_uri
    try:
        assert main(["models"]) == 0
    finally:
        del os.environ["MLFLOW_TRACKING_URI"]
    assert "v  1" in capsys.readouterr().out
    assert MlflowClient  # imported for type clarity
