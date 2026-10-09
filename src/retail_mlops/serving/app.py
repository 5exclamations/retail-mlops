"""FastAPI prediction service with Prometheus instrumentation and background monitoring."""
from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from retail_mlops import __version__
from retail_mlops.config import Config, load_config
from retail_mlops.db import get_engine, init_db, log_predictions, read_sales, upsert_sales
from retail_mlops.monitoring.metrics import Metrics
from retail_mlops.monitoring.service import build_report
from retail_mlops.registry import store
from retail_mlops.registry.bundle import ModelBundle, load_bundle
from retail_mlops.serving.predictor import ForecastError, forecast
from retail_mlops.serving.schemas import GroundTruthRequest, PredictRequest, PredictResponse


class AppState:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.engine = get_engine(cfg.database_url)
        self.metrics = Metrics()
        self.bundle: ModelBundle | None = None
        self.report: dict | None = None
        self.lock = threading.Lock()

    def set_bundle(self, bundle: ModelBundle | None) -> None:
        with self.lock:
            self.bundle = bundle
        if bundle:
            self.metrics.set_model(bundle.name, bundle.version, bundle.run_id, bundle.reference_wape)

    def try_load_champion(self) -> bool:
        try:
            self.set_bundle(load_bundle(self.cfg))
            return True
        except Exception:  # no champion registered yet / registry unreachable
            return False

    def champion_version(self) -> str | None:
        try:
            return store.alias_version(store.client_for(self.cfg), self.cfg, self.cfg.mlflow.champion_alias)
        except Exception:
            return None

    def refresh_monitoring(self) -> dict | None:
        if self.bundle is None:
            return None
        self.report = build_report(self.engine, self.bundle, self.cfg.monitoring)
        self.metrics.update_from_report(self.report)
        return self.report


async def _background(state: AppState) -> None:
    mon = state.cfg.monitoring
    last_refresh = last_check = 0.0
    while True:
        await asyncio.sleep(1)
        now = time.monotonic()
        try:
            if mon.reload_check_seconds and now - last_check >= mon.reload_check_seconds:
                last_check = now
                cur = state.champion_version()
                if cur and (state.bundle is None or cur != state.bundle.version):
                    await asyncio.to_thread(state.try_load_champion)
            if mon.refresh_seconds and now - last_refresh >= mon.refresh_seconds:
                last_refresh = now
                await asyncio.to_thread(state.refresh_monitoring)
        except Exception:
            state.metrics.errors.labels("background").inc()


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config(os.getenv("RETAIL_CONFIG"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = AppState(cfg)
        init_db(state.engine)
        state.try_load_champion()
        app.state.s = state
        task = None
        if cfg.monitoring.refresh_seconds or cfg.monitoring.reload_check_seconds:
            task = asyncio.create_task(_background(state))
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Retail Demand Forecast API", version=__version__, lifespan=lifespan)

    def st(request: Request) -> AppState:
        return request.app.state.s

    def require_admin(x_api_key: str | None = Header(None)) -> None:
        expected = os.getenv("ADMIN_API_KEY")
        if expected and x_api_key != expected:
            raise HTTPException(401, "invalid or missing X-API-Key")

    @app.middleware("http")
    async def instrument(request: Request, call_next):
        s: AppState | None = getattr(request.app.state, "s", None)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            endpoint = route.path if route else "unmatched"
            if s is not None and endpoint != "/metrics":
                s.metrics.requests.labels(endpoint, request.method, str(status)).inc()
                s.metrics.request_latency.labels(endpoint).observe(time.perf_counter() - t0)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready(request: Request):
        s = st(request)
        try:
            with s.engine.connect() as c:
                c.execute(text("SELECT 1"))
        except Exception as e:
            raise HTTPException(503, f"database unavailable: {e}") from e
        if s.bundle is None:
            raise HTTPException(503, "no champion model loaded")
        return {"status": "ready", "model_version": s.bundle.version}

    @app.get("/model")
    def model_info(request: Request):
        b = st(request).bundle
        if b is None:
            raise HTTPException(503, "no champion model loaded")
        md = b.metadata
        return {"name": b.name, "version": b.version, "run_id": b.run_id, "algorithm": md.get("selected_model"),
                "trained_at": md.get("trained_at"), "data_hash": md.get("data_hash"), "train_end": md.get("train_end"),
                "horizon_days": b.cfg.forecast.horizon, "test_metrics": md.get("metrics", {}).get("test"),
                "baselines_test": {k: v["test"] for k, v in md.get("baselines", {}).items()}}

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request):
        s = st(request)
        if s.bundle is None:
            s.metrics.errors.labels("no_model").inc()
            raise HTTPException(503, "no champion model loaded")
        bundle = s.bundle  # snapshot: a concurrent reload cannot change the model mid-request
        rid = uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        try:
            results, rows = forecast(s.engine, bundle, [i.model_dump() for i in body.instances], "api", rid)
        except ForecastError as e:
            s.metrics.errors.labels("validation").inc()
            raise HTTPException(422, {"message": "cannot score request", "problems": e.problems}) from e
        except Exception as e:
            s.metrics.errors.labels("internal").inc()
            raise HTTPException(500, f"prediction failed: {type(e).__name__}") from e
        s.metrics.inference_latency.observe(time.perf_counter() - t0)
        for r in results:
            s.metrics.prediction_value.observe(r["prediction"])
            if r["low_history"]:
                s.metrics.low_history.inc()
        s.metrics.predictions.labels(bundle.version, "api").inc(len(results))
        try:
            log_predictions(s.engine, rows)
        except Exception:
            s.metrics.errors.labels("logging").inc()  # never fail a prediction because logging failed
        return PredictResponse(request_id=rid, model_name=bundle.name, model_version=bundle.version, predictions=results)

    @app.post("/ground-truth", dependencies=[Depends(require_admin)])
    def ground_truth(body: GroundTruthRequest, request: Request):
        """Ingest actual sales; they become ground truth for earlier predictions and training data for retrains."""
        s = st(request)
        df = pd.DataFrame([r.model_dump() for r in body.records])
        df["date"] = pd.to_datetime(df["date"])
        existing = read_sales(s.engine, start=df["date"].min() - pd.Timedelta(days=70), end=df["date"].max(),
                              series=list(df[["store_id", "sku_id"]].drop_duplicates().itertuples(index=False, name=None)))
        meta = existing.groupby(["store_id", "sku_id"])["category"].first().to_dict()
        ex = existing.set_index(["store_id", "sku_id", "date"])
        errors = []
        for i, r in df.iterrows():
            cat = r["category"] if isinstance(r["category"], str) else meta.get((r["store_id"], r["sku_id"]))
            if cat is None:
                errors.append(f"{r['store_id']}/{r['sku_id']}: unknown series, 'category' required")
                continue
            df.at[i, "category"] = cat
            key = (r["store_id"], r["sku_id"], r["date"])
            for c, default in (("price", None), ("promo", 0)):
                if pd.isna(r[c]):
                    df.at[i, c] = ex.loc[key, c] if key in ex.index else default
        if errors:
            raise HTTPException(422, {"message": "invalid records", "problems": sorted(set(errors))})
        n = upsert_sales(s.engine, df)
        return {"upserted": n}

    @app.get("/monitoring/report")
    def monitoring_report(request: Request):
        s = st(request)
        if s.report is None:
            raise HTTPException(404, "no report yet; POST /monitoring/refresh")
        return s.report

    @app.post("/monitoring/refresh", dependencies=[Depends(require_admin)])
    def monitoring_refresh(request: Request):
        s = st(request)
        report = s.refresh_monitoring()
        if report is None:
            raise HTTPException(503, "no champion model loaded")
        return report

    @app.post("/admin/reload", dependencies=[Depends(require_admin)])
    def reload_model(request: Request):
        s = st(request)
        if not s.try_load_champion():
            raise HTTPException(503, "could not load champion")
        return {"model_version": s.bundle.version}

    @app.get("/metrics")
    def metrics(request: Request):
        return Response(generate_latest(st(request).metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        s = getattr(request.app.state, "s", None)
        if s:
            s.metrics.errors.labels("unhandled").inc()
        return JSONResponse({"detail": "internal error"}, status_code=500)

    return app


app = None  # created lazily: `uvicorn retail_mlops.serving.app:create_app --factory`
