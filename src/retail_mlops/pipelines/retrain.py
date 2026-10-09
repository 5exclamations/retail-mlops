"""Retraining pipeline: decide -> (validate -> preprocess -> features -> train -> evaluate -> gate -> promote)."""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from sqlalchemy.engine import Engine

from retail_mlops.config import Config
from retail_mlops.db import read_sales, record_retrain, update_retrain
from retail_mlops.models.train import run_training
from retail_mlops.monitoring.service import build_report
from retail_mlops.registry import store
from retail_mlops.registry.bundle import ModelBundle, load_bundle


def decide(cfg: Config, engine: Engine, bundle: ModelBundle | None, force: bool = False) -> tuple[list[str], dict]:
    """Return (reasons, monitoring_report). Empty reasons = no retrain needed."""
    if bundle is None:
        return ["bootstrap: no champion model"], {}
    reasons: list[str] = []
    if force:
        reasons.append("forced")
    md = bundle.metadata
    age = (datetime.now(UTC) - datetime.fromisoformat(md["trained_at"])).days
    if age >= cfg.retrain.max_model_age_days:
        reasons.append(f"schedule: model is {age} days old (max {cfg.retrain.max_model_age_days})")
    new = read_sales(engine, start=pd.Timestamp(md["data_end"]) + pd.Timedelta(days=1))
    if len(new) >= cfg.retrain.min_new_rows:
        reasons.append(f"data: {len(new)} new rows since training (min {cfg.retrain.min_new_rows})")
    report = build_report(engine, bundle, cfg.monitoring)
    if report["drift"].get("drifted"):
        alerts = [f["feature"] for f in report["drift"]["features"] if f["status"] == "alert"]
        reasons.append(f"drift: input drift on {alerts}")
    if report["performance"].get("degraded"):
        reasons.append(f"performance: live WAPE {report['performance']['wape']:.3f} is "
                       f"{report['performance']['wape_ratio']:.2f}x the reference")
    return reasons, report


def run_retraining(cfg: Config, engine: Engine, force: bool = False, dry_run: bool = False) -> dict:
    started = datetime.now(UTC).replace(tzinfo=None)
    store.client_for(cfg)
    try:
        bundle = load_bundle(cfg)
    except Exception:
        bundle = None
    reasons, report = decide(cfg, engine, bundle, force)
    summary: dict = {"reasons": reasons, "monitoring": report}
    if not reasons or dry_run:
        summary["status"] = "skipped" if not reasons else "dry_run"
        if not dry_run:
            record_retrain(engine, started_at=started, finished_at=datetime.now(UTC).replace(tzinfo=None),
                           trigger="none", status="skipped", details=summary)
        return summary

    rid = record_retrain(engine, started_at=started, trigger="; ".join(reasons), status="running", details={})
    try:
        raw = read_sales(engine)
        if cfg.retrain.training_window_days:
            raw = raw[raw["date"] > raw["date"].max() - pd.Timedelta(days=cfg.retrain.training_window_days)]
        raw = raw.assign(date=raw["date"].dt.strftime("%Y-%m-%d"))
        result = run_training(cfg, raw, trigger="retrain:" + reasons[0].split(":")[0])
        status = {"promoted": "promoted", "rejected": "rejected", "data_invalid": "failed"}.get(result.outcome, result.outcome)
        summary.update(status=status, outcome=result.outcome, mlflow_run_id=result.run_id, model_version=result.version,
                       gates=result.gates.to_dict() if result.gates else None,
                       validation_errors=[c.detail for c in result.validation.errors] if result.validation else None)
        update_retrain(engine, rid, finished_at=datetime.now(UTC).replace(tzinfo=None), status=status,
                       mlflow_run_id=result.run_id, model_version=result.version, details=summary)
    except Exception as e:
        summary.update(status="failed", error=f"{type(e).__name__}: {e}")
        update_retrain(engine, rid, finished_at=datetime.now(UTC).replace(tzinfo=None), status="failed", details=summary)
        raise
    return summary
