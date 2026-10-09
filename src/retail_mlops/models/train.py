"""Training + evaluation + gated promotion. One call = one reproducible, tracked experiment."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import sklearn

from retail_mlops.config import Config
from retail_mlops.data.preprocess import preprocess
from retail_mlops.data.validate import ValidationReport, validate_raw
from retail_mlops.features.engineering import (
    TARGET,
    build_features,
    feature_columns,
    time_split,
    training_frame,
)
from retail_mlops.models.estimators import make_baselines, make_candidates
from retail_mlops.models.metrics import compute_metrics, segment_metrics
from retail_mlops.registry import store
from retail_mlops.registry.bundle import load_bundle, log_bundle_files
from retail_mlops.registry.gates import GateResult, evaluate_gates


@dataclass
class TrainResult:
    outcome: str  # promoted | rejected | not_registered | data_invalid
    run_id: str | None = None
    version: str | None = None
    selected_model: str | None = None
    metrics: dict = field(default_factory=dict)
    gates: GateResult | None = None
    validation: ValidationReport | None = None


def fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a dataframe (order-insensitive on rows) for lineage."""
    d = df.sort_values(list(df.columns[:3])).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(d, index=False).values.tobytes()).hexdigest()[:16]


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).parent).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) else {key: str(v)[:480]})
    return out


def run_training(cfg: Config, raw: pd.DataFrame, trigger: str = "manual", register: bool = True) -> TrainResult:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    np.random.seed(cfg.seed)

    report = validate_raw(raw, cfg.validation)
    data_hash = fingerprint(raw)
    with mlflow.start_run(run_name=f"train-{datetime.now(UTC):%Y%m%d-%H%M%S}") as run:
        mlflow.set_tags({"trigger": trigger, "data_hash": data_hash, "git_sha": _git_sha(),
                         "python": platform.python_version(), "sklearn": sklearn.__version__})
        mlflow.log_params(_flat(cfg.model_dump()))
        mlflow.log_dict(report.to_dict(), "validation_report.json")
        if not report.passed:
            mlflow.set_tag("outcome", "data_invalid")
            return TrainResult("data_invalid", run.info.run_id, validation=report)

        panel, pre = preprocess(raw)
        mlflow.log_metrics({f"preprocess_{k}": v for k, v in pre.to_dict().items()})
        feats = build_features(panel, cfg.forecast)
        frame = training_frame(feats, cfg.forecast)
        splits = time_split(frame, cfg.forecast)
        cols = feature_columns(cfg.forecast)
        for name, s in splits.items():
            mlflow.log_params({f"{name}_rows": len(s), f"{name}_start": s["date"].min().date(), f"{name}_end": s["date"].max().date()})
        train, valid, test = splits["train"], splits["valid"], splits["test"]

        # ---- baselines + candidates; each gets a nested run with its own metrics
        results: dict[str, dict] = {}
        fitted: dict[str, object] = {}
        models = {**make_baselines(cfg), **make_candidates(cfg)}
        for name, est in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                est.fit(train[cols], train[TARGET])
                mv_, mt_ = (compute_metrics(s[TARGET], est.predict(s[cols])) for s in (valid, test))
                mlflow.set_tag("kind", "baseline" if name in make_baselines(cfg) else "candidate")
                mlflow.log_metrics({**{f"valid_{k}": v for k, v in mv_.items()}, **{f"test_{k}": v for k, v in mt_.items()}})
            results[name] = {"valid": mv_, "test": mt_}
            fitted[name] = est

        baseline_names, cand_names = list(make_baselines(cfg)), list(make_candidates(cfg))
        sel_metric = cfg.models.selection_metric
        selected = min(cand_names, key=lambda n: results[n]["valid"][sel_metric])  # selection on VALID only
        best_base = min(baseline_names, key=lambda n: results[n]["valid"][sel_metric])
        pipe = fitted[selected]
        mlflow.set_tags({"selected_model": selected, "best_baseline": best_base})
        mlflow.log_metrics({f"test_{k}": v for k, v in results[selected]["test"].items()})
        mlflow.log_metrics({f"valid_{k}": v for k, v in results[selected]["valid"].items()})
        mlflow.log_metrics({f"baseline_test_{k}": v for k, v in results[best_base]["test"].items() if k != "n"})

        # ---- the gate compares everything on the untouched test window
        test_pred = np.clip(pipe.predict(test[cols]), 0, None)
        base_pred = fitted[best_base].predict(test[cols])
        cand_metrics = compute_metrics(test[TARGET], test_pred)
        results[selected]["test"] = cand_metrics
        cand_seg = segment_metrics(test, test_pred, "category")
        base_seg = segment_metrics(test, base_pred, "category")
        champion_metrics, champ_version = _score_champion(cfg, panel, test)
        gates = evaluate_gates(cfg.gates, cand_metrics, results[best_base]["test"], cand_seg, base_seg,
                               champion_metrics, report.passed)
        mlflow.log_dict(gates.to_dict(), "gates.json")
        mlflow.log_dict({"candidate": cand_seg, "baseline": base_seg}, "segment_metrics.json")
        mlflow.log_metric("gates_passed", int(gates.passed))
        if champion_metrics:
            mlflow.log_metrics({"champion_test_wape": champion_metrics["wape"]})

        metadata = {
            "model_name": cfg.mlflow.model_name, "selected_model": selected, "best_baseline": best_base,
            "trained_at": datetime.now(UTC).isoformat(), "data_hash": data_hash, "seed": cfg.seed,
            "trigger": trigger, "git_sha": _git_sha(), "train_end": str(train["date"].max().date()),
            "data_end": str(panel["date"].max().date()), "n_train": len(train),
            "metrics": {"valid": results[selected]["valid"], "test": cand_metrics},
            "baselines": {n: results[n] for n in baseline_names},
            "all_models": results, "gates": gates.to_dict(), "champion_version_compared": champ_version,
            "preprocess": pre.to_dict(),
        }
        ref_rows = pd.concat([train, valid]).sample(min(cfg.models.reference_sample_size, len(train) + len(valid)),
                                                   random_state=cfg.seed)
        reference = ref_rows[cols].reset_index(drop=True)
        reference["__prediction"] = np.clip(pipe.predict(reference[cols]), 0, None)

        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")  # own registry: trusted artifacts
        log_bundle_files(cfg, metadata, reference)

        if not register:
            mlflow.set_tag("outcome", "not_registered")
            return TrainResult("not_registered", run.info.run_id, None, selected, metadata["metrics"], gates, report)

        client = store.client_for(cfg)
        version = store.register_version(client, cfg, run.info.run_id, {
            "gates_passed": str(gates.passed), "test_wape": f"{cand_metrics['wape']:.5f}",
            "data_hash": data_hash, "selected_model": selected, "stage": "candidate"})
        client.set_registered_model_alias(cfg.mlflow.model_name, cfg.mlflow.candidate_alias, version)
        if gates.passed:
            store.promote(client, cfg, version)
            outcome = "promoted"
        else:
            client.set_model_version_tag(cfg.mlflow.model_name, version, "stage", "rejected")
            outcome = "rejected"
        mlflow.set_tags({"outcome": outcome, "model_version": version})
        return TrainResult(outcome, run.info.run_id, version, selected, metadata["metrics"], gates, report)


def _score_champion(cfg: Config, panel: pd.DataFrame, test: pd.DataFrame):
    """Re-score the current champion on the new test window so the comparison is apples-to-apples."""
    client = store.client_for(cfg)
    version = store.alias_version(client, cfg, cfg.mlflow.champion_alias)
    if version is None:
        return None, None
    bundle = load_bundle(cfg, version=version)
    scored = bundle.score_panel(panel, test)
    scored = scored.dropna(subset=[TARGET])
    if scored.empty:
        return None, version
    return compute_metrics(scored[TARGET], scored["prediction"]), version


def write_report(result: TrainResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "outcome": result.outcome, "run_id": result.run_id, "version": result.version,
        "selected_model": result.selected_model, "metrics": result.metrics,
        "gates": result.gates.to_dict() if result.gates else None,
    }, indent=2, default=str))

