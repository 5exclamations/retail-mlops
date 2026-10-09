"""Training + evaluation + gated promotion. One call = one reproducible, tracked experiment.

Evaluation protocol (this is the part that matters for honest promotion decisions):

1. If a champion exists, the final test window starts the day AFTER the champion's last data date, so the
   champion has never seen it (not for fitting, not for selection). The challenger is fitted and selected only on
   data before that window. If fewer than ``gates.min_holdout_days`` fresh days exist, no fair comparison is
   possible and the run ends with outcome ``insufficient_holdout`` (nothing is registered).
2. Model selection uses walk-forward folds that sit immediately before the test window, never the test window.
3. The selected recipe is refit on all data before the test window and evaluated once on the test window.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.artifacts
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
    split_at,
    training_frame,
)
from retail_mlops.models.backtest import walk_forward
from retail_mlops.models.estimators import make_baselines, make_candidates
from retail_mlops.models.metrics import compute_metrics, segment_metrics
from retail_mlops.registry import store
from retail_mlops.registry.bundle import load_bundle, log_bundle_files
from retail_mlops.registry.gates import GateResult, evaluate_gates, paired_bootstrap
from retail_mlops.registry.integrity import TRUSTED_TYPES, dir_sha256


@dataclass
class TrainResult:
    outcome: str  # promoted | rejected | not_registered | data_invalid | insufficient_holdout
    run_id: str | None = None
    version: str | None = None
    selected_model: str | None = None
    metrics: dict = field(default_factory=dict)
    gates: GateResult | None = None
    validation: ValidationReport | None = None
    reasons: list[str] = field(default_factory=list)


def fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a dataframe (row-order insensitive) for dataset versioning/lineage."""
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
    f = cfg.forecast

    report = validate_raw(raw, cfg.validation)
    data_hash = fingerprint(raw)
    with mlflow.start_run(run_name=f"train-{datetime.now(UTC):%Y%m%d-%H%M%S}") as run:
        rid = run.info.run_id
        mlflow.set_tags({"trigger": trigger, "data_hash": data_hash, "git_sha": _git_sha(),
                         "python": platform.python_version(), "sklearn": sklearn.__version__})
        mlflow.log_params(_flat(cfg.model_dump()))
        mlflow.log_dict(report.to_dict(), "validation_report.json")
        if not report.passed:
            mlflow.set_tag("outcome", "data_invalid")
            return TrainResult("data_invalid", rid, validation=report, reasons=[c.detail for c in report.errors])

        panel, pre_rep = preprocess(raw)
        mlflow.log_metrics({f"preprocess_{k}": v for k, v in pre_rep.to_dict().items()})
        frame = training_frame(build_features(panel, f), f)
        cols = feature_columns(f)
        last = frame["date"].max()

        # ---- champion + fresh, comparable holdout
        champ_bundle, champ_version = None, None
        if register:
            client = store.client_for(cfg)
            champ_version = store.alias_version(client, cfg, cfg.mlflow.champion_alias)
            champ_bundle = load_bundle(cfg, version=champ_version) if champ_version else None
        test_start = last - pd.Timedelta(days=f.test_days - 1)
        if champ_bundle is not None:
            test_start = max(test_start, pd.Timestamp(champ_bundle.metadata["data_end"]) + pd.Timedelta(days=1))
        holdout_days = int((last - test_start).days + 1)
        mlflow.log_params({"test_start": test_start.date(), "test_end": last.date(), "holdout_days": holdout_days})
        if champ_bundle is not None and holdout_days < cfg.gates.min_holdout_days:
            why = (f"only {holdout_days} days of data after champion v{champ_version}'s data end "
                   f"({champ_bundle.metadata['data_end']}); need >= {cfg.gates.min_holdout_days} for a fair comparison")
            mlflow.set_tags({"outcome": "insufficient_holdout", "rejection_reasons": why})
            return TrainResult("insufficient_holdout", rid, validation=report, reasons=[why])
        pre, test = split_at(frame, test_start)

        # ---- walk-forward selection (pre-test data only) + final fit
        baselines, candidates = make_baselines(cfg), make_candidates(cfg)
        backtests, results, fitted = {}, {}, {}
        for name, est in {**baselines, **candidates}.items():
            with mlflow.start_run(run_name=name, nested=True):
                bt = walk_forward(pre, cols, est, test_start, f.backtest_folds, f.backtest_step_days)
                est.fit(pre[cols], pre[TARGET])
                tm = compute_metrics(test[TARGET], np.clip(est.predict(test[cols]), 0, None))
                mlflow.set_tag("kind", "baseline" if name in baselines else "candidate")
                mlflow.log_metrics({"backtest_wape_mean": bt["mean_wape"], "backtest_wape_std": bt["std_wape"],
                                    **{f"test_{k}": v for k, v in tm.items()}})
            backtests[name], results[name], fitted[name] = bt, tm, est
        sel_key = lambda n: backtests[n]["mean_wape"]  # noqa: E731  (selection NEVER looks at test metrics)
        selected, best_base = min(candidates, key=sel_key), min(baselines, key=sel_key)
        pipe = fitted[selected]
        test_pred = np.clip(pipe.predict(test[cols]), 0, None)
        base_pred = np.clip(fitted[best_base].predict(test[cols]), 0, None)
        cand_metrics = compute_metrics(test[TARGET], test_pred)
        mlflow.set_tags({"selected_model": selected, "best_baseline": best_base})
        mlflow.log_metrics({**{f"test_{k}": v for k, v in cand_metrics.items()},
                            "backtest_wape_mean": backtests[selected]["mean_wape"],
                            "baseline_test_wape": results[best_base]["wape"]})

        # ---- breakdowns on the test window
        breakdowns = {
            by: {"candidate": segment_metrics(test, test_pred, by), "baseline": segment_metrics(test, base_pred, by)}
            for by in ("category", "store_id", "sku_id")}
        mlflow.log_dict(breakdowns, "breakdowns_test.json")
        mlflow.log_dict({n: {k: v for k, v in b.items()} for n, b in backtests.items()}, "backtest.json")

        # ---- champion on the SAME fresh rows, paired comparison
        champ_metrics, paired = None, None
        if champ_bundle is not None:
            scored = champ_bundle.score_panel(panel, test[["store_id", "sku_id", "date"]])
            cmp_df = test[["store_id", "sku_id", "date", TARGET]].assign(p_candidate=test_pred).merge(
                scored[["store_id", "sku_id", "date", "prediction"]].rename(columns={"prediction": "p_champion"}),
                on=["store_id", "sku_id", "date"], how="inner").rename(columns={TARGET: "y"})
            champ_metrics = compute_metrics(cmp_df["y"], cmp_df["p_champion"])
            cand_metrics = compute_metrics(cmp_df["y"], cmp_df["p_candidate"])
            paired = paired_bootstrap(cmp_df, cfg.gates.bootstrap_samples, cfg.gates.significance_level, cfg.seed)
            mlflow.log_metrics({"champion_test_wape": champ_metrics["wape"],
                                "improvement_vs_champion": paired["rel_improvement"],
                                "improvement_ci_low": paired["ci_low"], "improvement_ci_high": paired["ci_high"]})
        gates = evaluate_gates(cfg.gates, cand_metrics, results[best_base],
                               breakdowns["category"]["candidate"], breakdowns["category"]["baseline"],
                               champ_metrics, report.passed, paired, holdout_days)
        mlflow.log_dict(gates.to_dict(), "gates.json")
        mlflow.log_metric("gates_passed", int(gates.passed))

        # ---- model + bundle (skops format; hash recorded for integrity checks)
        info = mlflow.sklearn.log_model(pipe, name="model", serialization_format="skops",
                                        skops_trusted_types=TRUSTED_TYPES)
        with tempfile.TemporaryDirectory() as d:
            sha = dir_sha256(mlflow.artifacts.download_artifacts(artifact_uri=info.model_uri, dst_path=d))
        metadata = {
            "model_name": cfg.mlflow.model_name, "selected_model": selected, "best_baseline": best_base,
            "trained_at": datetime.now(UTC).isoformat(), "data_hash": data_hash, "seed": cfg.seed,
            "trigger": trigger, "git_sha": _git_sha(), "model_sha256": sha,
            "train_end": str(pre["date"].max().date()), "test_start": str(test_start.date()),
            "data_end": str(panel["date"].max().date()), "n_train": int(len(pre)), "holdout_days": holdout_days,
            "metrics": {"backtest": {k: backtests[selected][k] for k in ("mean_wape", "std_wape", "pooled")},
                        "test": cand_metrics},
            "baselines": {n: {"test": results[n], "backtest_wape": backtests[n]["mean_wape"]} for n in baselines},
            "all_models_test": results, "by_lead_backtest": backtests[selected]["by_lead"],
            "gates": gates.to_dict(), "champion_version_compared": champ_version, "champion_test": champ_metrics,
            "paired_vs_champion": paired, "preprocess": pre_rep.to_dict(),
        }
        ref_rows = pre.sample(min(cfg.models.reference_sample_size, len(pre)), random_state=cfg.seed)
        reference = ref_rows[cols].reset_index(drop=True)
        reference["__prediction"] = np.clip(pipe.predict(reference[cols]), 0, None)
        log_bundle_files(cfg, metadata, reference)
        result = lambda outcome, version=None: TrainResult(  # noqa: E731
            outcome, rid, version, selected, metadata["metrics"], gates, report, gates.reasons)

        if not register:
            mlflow.set_tag("outcome", "not_registered")
            return result("not_registered")

        version = store.register_version(client, cfg, rid, {
            "gates_passed": str(gates.passed), "test_wape": f"{cand_metrics['wape']:.5f}", "data_hash": data_hash,
            "selected_model": selected, "stage": "candidate", "model_sha256": sha, "data_end": metadata["data_end"],
            "test_start": metadata["test_start"], "git_sha": metadata["git_sha"],
            "rejection_reasons": " | ".join(gates.reasons)[:4000]})
        client.set_registered_model_alias(cfg.mlflow.model_name, cfg.mlflow.candidate_alias, version)
        if gates.passed:
            store.promote(client, cfg, version)
            outcome = "promoted"
        else:
            client.set_model_version_tag(cfg.mlflow.model_name, version, "stage", "rejected")
            outcome = "rejected"
        mlflow.set_tags({"outcome": outcome, "model_version": version, "rejection_reasons": " | ".join(gates.reasons)[:4000]})
        return result(outcome, version)


def write_report(result: TrainResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "outcome": result.outcome, "run_id": result.run_id, "version": result.version,
        "selected_model": result.selected_model, "metrics": result.metrics, "reasons": result.reasons,
        "gates": result.gates.to_dict() if result.gates else None,
    }, indent=2, default=str))
