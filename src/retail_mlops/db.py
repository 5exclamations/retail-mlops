"""SQLAlchemy persistence: sales history (feature source + ground truth), prediction log, retrain runs.

Works on PostgreSQL (docker-compose, production-like) and SQLite (unit tests, zero-setup local runs)."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
    text,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

sales_history = Table(
    "sales_history", metadata,
    Column("store_id", String(16), primary_key=True),
    Column("sku_id", String(16), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("category", String(64), nullable=False),
    Column("units", Float),
    Column("price", Float),
    Column("promo", Integer),
)

prediction_log = Table(
    "prediction_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime, nullable=False, default=lambda: datetime.now(UTC).replace(tzinfo=None)),
    Column("request_id", String(64)),
    Column("source", String(16), nullable=False),  # "api" | "batch"
    Column("model_name", String(128)),
    Column("model_version", String(32), nullable=False),
    Column("store_id", String(16), nullable=False),
    Column("sku_id", String(16), nullable=False),
    Column("target_date", Date, nullable=False),
    Column("prediction", Float, nullable=False),
    Column("horizon", Integer),  # days between last observed sales and target_date; <=0 = retrospective
    Column("features", Text),  # JSON of the model inputs, used for drift monitoring
    Index("ix_pred_series_date", "store_id", "sku_id", "target_date"),
    Index("ix_pred_created", "created_at"),
)

retrain_runs = Table(
    "retrain_runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("started_at", DateTime, nullable=False),
    Column("finished_at", DateTime),
    Column("trigger", String(64), nullable=False),
    Column("status", String(32), nullable=False),  # running | promoted | rejected | skipped | failed
    Column("mlflow_run_id", String(64)),
    Column("model_version", String(32)),
    Column("details", Text),
)


def get_engine(url: str) -> Engine:
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    return create_engine(url, **kwargs)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)


def _upsert_stmt(engine: Engine, table: Table):
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    stmt = insert(table)
    keys = [c.name for c in table.primary_key.columns]
    update = {c.name: stmt.excluded[c.name] for c in table.columns if c.name not in keys}
    return stmt.on_conflict_do_update(index_elements=keys, set_=update)


def upsert_sales(engine: Engine, df: pd.DataFrame, chunk: int = 5000) -> int:
    """Insert or update (store, sku, date) rows. Used for bulk loads and for late-arriving ground truth."""
    cols = [c.name for c in sales_history.columns]
    d = df[cols].copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d = d.astype(object).where(d.notna(), None)
    records = d.to_dict("records")
    stmt = _upsert_stmt(engine, sales_history)
    with engine.begin() as conn:
        for i in range(0, len(records), chunk):
            conn.execute(stmt, records[i : i + chunk])
    return len(records)


def read_sales(engine: Engine, start=None, end=None, series: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    q = select(sales_history)
    if start is not None:
        q = q.where(sales_history.c.date >= pd.Timestamp(start).date())
    if end is not None:
        q = q.where(sales_history.c.date <= pd.Timestamp(end).date())
    if series:
        q = q.where(sales_history.c.store_id.in_({s for s, _ in series}), sales_history.c.sku_id.in_({k for _, k in series}))
    df = pd.read_sql(q, engine)
    df["date"] = pd.to_datetime(df["date"])
    if series:
        wanted = set(series)
        mask = [(a, b) in wanted for a, b in zip(df["store_id"], df["sku_id"], strict=True)]
        df = df[pd.Series(mask, index=df.index, dtype=bool)]
    return df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)


def last_sales_date(engine: Engine) -> pd.Timestamp | None:
    with engine.connect() as c:
        v = c.execute(text("SELECT MAX(date) FROM sales_history")).scalar()
    return pd.Timestamp(v) if v else None


def log_predictions(engine: Engine, rows: list[dict]) -> None:
    if not rows:
        return
    now = datetime.now(UTC).replace(tzinfo=None)
    recs = [{**r, "created_at": r.get("created_at", now),
             "features": json.dumps(r["features"]) if not isinstance(r.get("features"), str) else r["features"]} for r in rows]
    with engine.begin() as conn:
        conn.execute(prediction_log.insert(), recs)


def read_prediction_log(engine: Engine, since: datetime | None = None, model_version: str | None = None) -> pd.DataFrame:
    q = select(prediction_log)
    if since is not None:
        q = q.where(prediction_log.c.created_at >= since)
    if model_version is not None:
        q = q.where(prediction_log.c.model_version == str(model_version))
    df = pd.read_sql(q, engine)
    if not df.empty:
        df["target_date"] = pd.to_datetime(df["target_date"])
    return df


def record_retrain(engine: Engine, **kw) -> int:
    kw["details"] = json.dumps(kw.get("details", {}), default=str)
    with engine.begin() as conn:
        return conn.execute(retrain_runs.insert().values(**kw)).inserted_primary_key[0]


def update_retrain(engine: Engine, run_id: int, **kw) -> None:
    if "details" in kw:
        kw["details"] = json.dumps(kw["details"], default=str)
    with engine.begin() as conn:
        conn.execute(retrain_runs.update().where(retrain_runs.c.id == run_id).values(**kw))


def retrain_summary(engine: Engine) -> list[dict]:
    """Counts and latest finish time per status, for Prometheus."""
    with engine.connect() as c:
        rows = c.execute(select(retrain_runs.c.status, retrain_runs.c.finished_at)).all()
    agg: dict[str, dict] = {}
    for status, fin in rows:
        a = agg.setdefault(status, {"status": status, "n": 0, "last_finished_ts": None})
        a["n"] += 1
        if fin is not None:
            ts = fin.replace(tzinfo=UTC).timestamp()
            a["last_finished_ts"] = max(a["last_finished_ts"] or 0, ts)
    return list(agg.values())


def read_retrain_history(engine: Engine, limit: int = 20) -> list[dict]:
    with engine.connect() as c:
        rows = c.execute(select(retrain_runs).order_by(retrain_runs.c.id.desc()).limit(limit)).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        d["details"] = json.loads(d["details"]) if d.get("details") else {}
        d["details"].pop("monitoring", None)
        out.append({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in d.items()})
    return out
