"""Input / prediction drift statistics: PSI (all features), KS test (numeric features)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from retail_mlops.config import MonitoringCfg

EPS = 1e-4


def _psi_from_counts(ref: np.ndarray, cur: np.ndarray) -> float:
    r = np.clip(ref / max(ref.sum(), 1), EPS, None)
    c = np.clip(cur / max(cur.sum(), 1), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with bin edges taken from reference quantiles. Open-ended outer bins mean
    values far outside the reference range still register as drift."""
    ref, cur = ref.dropna().to_numpy(float), cur.dropna().to_numpy(float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant feature: use a two-bin split on its value
        edges = np.array([ref.min() - 1, ref.min() + 1e-9, ref.max() + 1]) if ref.min() == ref.max() else edges
    inner = edges[1:-1]
    rc = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(inner) + 1)
    cc = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(inner) + 1)
    return _psi_from_counts(rc, cc)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna().astype(str)) | set(cur.dropna().astype(str)))
    rc = ref.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()
    cc = cur.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()
    return _psi_from_counts(rc, cc)


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float | None
    ks_pvalue: float | None
    status: str  # ok | warn | alert


def _status(psi: float, ks_p: float | None, cfg: MonitoringCfg) -> str:
    if np.isnan(psi):
        return "ok"
    if psi >= cfg.psi_alert and (ks_p is None or ks_p < cfg.ks_pvalue):
        return "alert"  # large shift AND statistically significant
    if psi >= cfg.psi_warn:
        return "warn"
    return "ok"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str], cfg: MonitoringCfg) -> dict:
    """Compare live model inputs with the training reference. Returns a JSON-friendly dict."""
    if len(current) < cfg.min_samples:
        return {"status": "insufficient_data", "n_current": int(len(current)), "min_samples": cfg.min_samples,
                "features": [], "drift_share": 0.0, "drifted": False}
    out: list[FeatureDrift] = []
    for f in features:
        if f not in reference or f not in current:
            continue
        r, c = reference[f], current[f]
        if pd.api.types.is_numeric_dtype(r):
            psi = psi_numeric(r, c)
            rr, cc = r.dropna(), c.dropna()
            ks = stats.ks_2samp(rr, cc) if len(rr) and len(cc) else None
            out.append(FeatureDrift(f, psi, float(ks.statistic) if ks else None, float(ks.pvalue) if ks else None,
                                    _status(psi, float(ks.pvalue) if ks else None, cfg)))
        else:
            psi = psi_categorical(r, c)
            out.append(FeatureDrift(f, psi, None, None, _status(psi, None, cfg)))
    n_alert = sum(1 for d in out if d.status == "alert")
    share = n_alert / len(out) if out else 0.0
    return {"status": "ok", "n_current": int(len(current)), "features": [asdict(d) for d in out],
            "drift_share": share, "drifted": share >= cfg.drift_share_alert}


def prediction_drift(ref_pred: pd.Series, cur_pred: pd.Series, cfg: MonitoringCfg) -> dict:
    if len(cur_pred) < cfg.min_samples:
        return {"status": "insufficient_data", "psi": None}
    psi = psi_numeric(ref_pred, cur_pred)
    ks = stats.ks_2samp(ref_pred.dropna(), cur_pred.dropna())
    return {"status": _status(psi, float(ks.pvalue), cfg), "psi": psi, "ks_pvalue": float(ks.pvalue),
            "ref_mean": float(ref_pred.mean()), "cur_mean": float(cur_pred.mean()),
            "ref_p50": float(ref_pred.median()), "cur_p50": float(cur_pred.median()),
            "ref_p95": float(ref_pred.quantile(0.95)), "cur_p95": float(cur_pred.quantile(0.95))}
