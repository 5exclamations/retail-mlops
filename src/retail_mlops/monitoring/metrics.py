"""Prometheus instruments. One Metrics object per app (own registry) so tests can build many apps."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("retail_http_requests_total", "HTTP requests", ["endpoint", "method", "status"], registry=r)
        self.request_latency = Histogram("retail_http_request_duration_seconds", "End-to-end request latency",
                                         ["endpoint"], buckets=LATENCY_BUCKETS, registry=r)
        self.inference_latency = Histogram("retail_model_inference_seconds", "Feature build + model inference time",
                                           buckets=LATENCY_BUCKETS, registry=r)
        self.predictions = Counter("retail_predictions_total", "Predictions served", ["model_version", "source"], registry=r)
        self.prediction_value = Histogram("retail_prediction_units", "Distribution of predicted units",
                                          buckets=(0, 1, 2, 5, 10, 20, 35, 50, 75, 100, 150, 250, 500), registry=r)
        self.errors = Counter("retail_prediction_errors_total", "Failed prediction requests", ["type"], registry=r)
        self.low_history = Counter("retail_low_history_predictions_total", "Predictions on series with short history", registry=r)

        self.model_info = Gauge("retail_model_info", "Currently served model", ["model_name", "version", "run_id"], registry=r)
        self.model_reference_wape = Gauge("retail_model_reference_wape", "WAPE on the held-out test set at training time", registry=r)

        self.feature_psi = Gauge("retail_feature_psi", "PSI of live input feature vs training reference", ["feature"], registry=r)
        self.feature_alert = Gauge("retail_feature_drift_alert", "1 if feature is in alert state", ["feature"], registry=r)
        self.drift_share = Gauge("retail_drift_share", "Share of monitored features in alert state", registry=r)
        self.drifted = Gauge("retail_input_drift_detected", "1 if overall input drift is flagged", registry=r)
        self.prediction_psi = Gauge("retail_prediction_psi", "PSI of live prediction distribution vs reference", registry=r)

        self.live_wape = Gauge("retail_live_wape", "WAPE over recent predictions with ground truth", registry=r)
        self.live_mae = Gauge("retail_live_mae", "MAE over recent predictions with ground truth", registry=r)
        self.live_bias = Gauge("retail_live_bias", "Relative bias over recent predictions with ground truth", registry=r)
        self.gt_coverage = Gauge("retail_ground_truth_coverage", "Share of recent predictions that have actuals", registry=r)
        self.gt_samples = Gauge("retail_ground_truth_samples", "Recent predictions with actuals", registry=r)
        self.live_wape_by_horizon = Gauge("retail_live_wape_by_horizon", "Live WAPE per forecast horizon (days)", ["horizon"], registry=r)
        self.pending = Gauge("retail_pending_predictions", "Recent predictions whose actuals have not matured", registry=r)
        self.model_trained_ts = Gauge("retail_model_trained_timestamp_seconds", "Training time of the served model", registry=r)
        self.monitor_refresh_ts = Gauge("retail_monitoring_last_refresh_timestamp_seconds", "Last successful monitoring refresh", registry=r)
        self.retrain_runs = Gauge("retail_retrain_runs", "Retraining runs recorded, by final status", ["status"], registry=r)
        self.retrain_last_ts = Gauge("retail_retrain_last_finished_timestamp_seconds", "Finish time of the latest retrain run", ["status"], registry=r)
        self.perf_degraded = Gauge("retail_performance_degraded", "1 if live WAPE exceeds the degradation threshold", registry=r)

    def set_model(self, name: str, version: str, run_id: str, reference_wape: float | None,
                  trained_at_ts: float | None = None) -> None:
        self.model_info.clear()
        if trained_at_ts:
            self.model_trained_ts.set(trained_at_ts)
        self.model_info.labels(name, str(version), run_id).set(1)
        self.model_reference_wape.set(reference_wape if reference_wape is not None else float("nan"))

    def update_from_report(self, report: dict) -> None:
        d = report.get("drift", {})
        for f in d.get("features", []):
            self.feature_psi.labels(f["feature"]).set(f["psi"])
            self.feature_alert.labels(f["feature"]).set(1 if f["status"] == "alert" else 0)
        if d.get("status") == "ok":
            self.drift_share.set(d["drift_share"])
            self.drifted.set(1 if d["drifted"] else 0)
        p = report.get("prediction_drift", {})
        if p.get("psi") is not None:
            self.prediction_psi.set(p["psi"])
        perf = report.get("performance", {})
        self.pending.set(perf.get("n_pending", 0))
        self.gt_coverage.set(perf.get("coverage", 0.0))
        for h, m in perf.get("by_horizon", {}).items():
            self.live_wape_by_horizon.labels(h).set(m["wape"])
        self.gt_samples.set(perf.get("n_with_ground_truth", 0))
        if perf.get("status") == "ok":
            self.live_wape.set(perf["wape"])
            self.live_mae.set(perf["mae"])
            self.live_bias.set(perf["bias"])
            self.perf_degraded.set(1 if perf.get("degraded") else 0)

    def update_retrains(self, rows: list[dict]) -> None:
        """rows: [{status, n, last_finished_ts}] aggregated from the retrain_runs table."""
        self.retrain_runs.clear()
        for r in rows:
            self.retrain_runs.labels(r["status"]).set(r["n"])
            if r.get("last_finished_ts"):
                self.retrain_last_ts.labels(r["status"]).set(r["last_finished_ts"])
