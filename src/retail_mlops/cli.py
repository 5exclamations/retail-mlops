"""Command line entry point: python -m retail_mlops <command>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from retail_mlops.config import DEFAULT_CONFIG_PATH, Config, load_config


def _cfg(args) -> Config:
    return load_config(DEFAULT_CONFIG_PATH, *(args.config or []))


def _engine(cfg: Config):
    from retail_mlops.db import get_engine, init_db

    eng = get_engine(cfg.database_url)
    init_db(eng)
    return eng


def cmd_generate(args) -> int:
    from retail_mlops.data.generate import DriftSpec, generate_sales

    cfg = _cfg(args)
    drift = DriftSpec(args.drift_start) if args.drift_start else None
    df = generate_sales(cfg, n_days=args.days, drift=drift)
    out = Path(args.out or cfg.data.raw_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows ({df['date'].min().date()} .. {df['date'].max().date()}) to {out}")
    return 0


def cmd_validate(args) -> int:
    from retail_mlops.data.validate import validate_raw

    cfg = _cfg(args)
    rep = validate_raw(pd.read_csv(args.input or cfg.data.raw_path), cfg.validation)
    print(rep.summary())
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep.to_dict(), indent=2))
    return 0 if rep.passed else 1


def cmd_ingest(args) -> int:
    from retail_mlops.data.preprocess import preprocess
    from retail_mlops.db import upsert_sales

    cfg = _cfg(args)
    raw = pd.read_csv(args.input or cfg.data.raw_path, parse_dates=["date"])
    if args.start:
        raw = raw[raw["date"] >= args.start]
    if args.end:
        raw = raw[raw["date"] <= args.end]
    clean, rep = preprocess(raw)  # the DB stores cleaned, gap-free history
    n = upsert_sales(_engine(cfg), clean)
    print(f"upserted {n:,} rows ({clean['date'].min().date()} .. {clean['date'].max().date()}); preprocess: {rep.to_dict()}")
    return 0


def cmd_train(args) -> int:
    from retail_mlops.db import read_sales
    from retail_mlops.models.train import run_training, write_report

    cfg = _cfg(args)
    if args.source == "db":
        raw = read_sales(_engine(cfg))
        raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
    else:
        raw = pd.read_csv(args.input or cfg.data.raw_path)
    res = run_training(cfg, raw, trigger="cli", register=not args.no_register)
    print(f"outcome={res.outcome} run_id={res.run_id} version={res.version} model={res.selected_model}")
    if res.validation and not res.validation.passed:
        print(res.validation.summary())
    if res.gates:
        print("gates:\n" + res.gates.summary())
    if res.metrics:
        print("selected-model metrics:", json.dumps(res.metrics, indent=2))
    if args.report:
        write_report(res, args.report)
    if args.require_promotion and res.outcome not in ("promoted", "not_registered"):
        return 2
    if args.require_promotion and res.gates and not res.gates.passed:
        return 2
    return 0 if res.outcome != "data_invalid" else 1


def cmd_batch(args) -> int:
    from retail_mlops.pipelines.batch_predict import run_batch
    from retail_mlops.registry.bundle import load_bundle

    cfg = _cfg(args)
    plan = pd.read_csv(args.plan) if args.plan else None
    df = run_batch(_engine(cfg), load_bundle(cfg), days=args.days, output=args.output, plan=plan)
    print(f"batch forecast: {len(df):,} rows, {df['date'].min()} .. {df['date'].max()} -> {args.output or '(not saved)'}")
    return 0


def cmd_monitor(args) -> int:
    from retail_mlops.monitoring.service import build_report
    from retail_mlops.registry.bundle import load_bundle

    cfg = _cfg(args)
    rep = build_report(_engine(cfg), load_bundle(cfg), cfg.monitoring)
    print(json.dumps(rep, indent=2, default=str))
    return 0


def cmd_retrain(args) -> int:
    from retail_mlops.pipelines.retrain import run_retraining

    cfg = _cfg(args)
    s = run_retraining(cfg, _engine(cfg), force=args.force, dry_run=args.dry_run)
    s.pop("monitoring", None)
    print(json.dumps(s, indent=2, default=str))
    return 1 if s.get("status") == "failed" else 0


def cmd_rollback(args) -> int:
    from retail_mlops.registry import store

    cfg = _cfg(args)
    v = store.rollback(store.client_for(cfg), cfg)
    print(f"champion now points to version {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="retail_mlops")
    p.add_argument("--config", action="append", help="extra YAML merged over configs/default.yaml (repeatable)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("generate", help="generate synthetic sales")
    s.add_argument("--out"); s.add_argument("--days", type=int); s.add_argument("--drift-start")
    s.set_defaults(fn=cmd_generate)
    s = sub.add_parser("validate", help="validate a raw CSV (exit 1 on failure)")
    s.add_argument("--input"); s.add_argument("--report")
    s.set_defaults(fn=cmd_validate)
    s = sub.add_parser("ingest", help="clean a raw CSV and load it into the sales_history table")
    s.add_argument("--input"); s.add_argument("--start"); s.add_argument("--end")
    s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("train", help="train, evaluate, gate and (maybe) promote")
    s.add_argument("--source", choices=["csv", "db"], default="csv"); s.add_argument("--input")
    s.add_argument("--no-register", action="store_true"); s.add_argument("--report")
    s.add_argument("--require-promotion", action="store_true", help="exit 2 if the candidate fails its gates (CI)")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("batch-predict", help="forecast next days for every series")
    s.add_argument("--days", type=int); s.add_argument("--output"); s.add_argument("--plan")
    s.set_defaults(fn=cmd_batch)
    s = sub.add_parser("monitor", help="print the drift / performance report")
    s.set_defaults(fn=cmd_monitor)
    s = sub.add_parser("retrain", help="retrain if a trigger fires")
    s.add_argument("--force", action="store_true"); s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_retrain)
    s = sub.add_parser("rollback", help="point champion back at the previous version")
    s.set_defaults(fn=cmd_rollback)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
