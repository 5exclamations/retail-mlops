"""Automated promotion gates. Pure functions, so they are trivially unit-testable."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from retail_mlops.config import GatesCfg


@dataclass
class GateCheck:
    name: str
    passed: bool
    value: float | None
    threshold: float | None
    detail: str


@dataclass
class GateResult:
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def reasons(self) -> list[str]:
        """Human-readable reasons for rejection (empty when promoted)."""
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "reasons": self.reasons, "checks": [asdict(c) for c in self.checks]}

    def summary(self) -> str:
        return "\n".join(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks)


def paired_bootstrap(df: pd.DataFrame, n: int, alpha: float, seed: int) -> dict:
    """Paired block bootstrap over DATES (days are resampled with all their series, preserving cross-series
    correlation). df needs: date, y, p_champion, p_candidate. Returns the relative WAPE improvement of the
    candidate over the champion with a (1-alpha) interval; positive = candidate better."""
    g = df.assign(ae_ch=(df["p_champion"] - df["y"]).abs(), ae_ca=(df["p_candidate"] - df["y"]).abs()).groupby("date")[
        ["ae_ch", "ae_ca", "y"]].sum()
    ae_ch, ae_ca = g["ae_ch"].to_numpy(), g["ae_ca"].to_numpy()
    point = 1 - ae_ca.sum() / ae_ch.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    boot = 1 - ae_ca[idx].sum(1) / ae_ch[idx].sum(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"rel_improvement": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_days": int(len(g))}


def evaluate_gates(
    cfg: GatesCfg,
    candidate: dict[str, float],
    best_baseline: dict[str, float],
    candidate_segments: dict[str, dict[str, float]],
    baseline_segments: dict[str, dict[str, float]],
    champion: dict[str, float] | None = None,
    data_valid: bool = True,
    paired: dict | None = None,
    holdout_days: int | None = None,
) -> GateResult:
    """All metric dicts are computed on the SAME fresh held-out window: dates after both models' training data.
    `paired` is the output of paired_bootstrap (required when a champion exists)."""
    res = GateResult()
    add = res.checks.append

    if cfg.require_valid_data:
        add(GateCheck("data_validation", data_valid, None, None,
                      "raw data passed validation" if data_valid else "raw data FAILED validation"))

    cw, bw = candidate["wape"], best_baseline["wape"]
    improvement = (bw - cw) / bw if bw > 0 else 0.0
    add(GateCheck("beats_baseline", improvement >= cfg.min_improvement_vs_baseline, improvement,
                  cfg.min_improvement_vs_baseline,
                  f"candidate WAPE {cw:.4f} vs best baseline {bw:.4f} ({improvement:+.1%} relative, "
                  f"need >= {cfg.min_improvement_vs_baseline:.1%})"))

    bias = abs(candidate["bias"])
    add(GateCheck("bias_within_bound", bias <= cfg.max_abs_bias, bias, cfg.max_abs_bias,
                  f"|bias| {bias:.4f} (max {cfg.max_abs_bias})"))

    worst, worst_seg = -1.0, ""
    for seg, m in candidate_segments.items():
        b = baseline_segments.get(seg)
        if not b or b["wape"] <= 0:
            continue
        rel = (m["wape"] - b["wape"]) / b["wape"]
        if rel > worst:
            worst, worst_seg = rel, seg
    add(GateCheck("no_segment_regression", worst <= cfg.max_segment_regression_vs_baseline,
                  worst if worst_seg else None, cfg.max_segment_regression_vs_baseline,
                  f"worst category vs baseline: {worst_seg or 'n/a'} ({worst:+.1%}, "
                  f"max {cfg.max_segment_regression_vs_baseline:+.1%})"))

    if champion is None:
        add(GateCheck("improves_on_champion", True, None, None,
                      "no champion yet; the first model only needs to clear the other gates"))
        return res

    add(GateCheck("fresh_holdout", (holdout_days or 0) >= cfg.min_holdout_days, holdout_days, cfg.min_holdout_days,
                  f"{holdout_days} fresh days after the champion's data (need >= {cfg.min_holdout_days})"))
    rel = 1 - cw / champion["wape"]
    add(GateCheck("improves_on_champion", rel >= cfg.min_improvement_vs_champion, rel, cfg.min_improvement_vs_champion,
                  f"candidate WAPE {cw:.4f} vs champion {champion['wape']:.4f} on the same fresh holdout "
                  f"({rel:+.2%} relative; need >= {cfg.min_improvement_vs_champion:.2%})"))
    if cfg.require_significance:
        ok = paired is not None and paired["ci_low"] > 0
        d = (f"improvement {paired['rel_improvement']:+.2%}, {1 - cfg.significance_level:.0%} CI "
             f"[{paired['ci_low']:+.2%}, {paired['ci_high']:+.2%}] over {paired['n_days']} days; CI must exclude 0"
             if paired else "no paired comparison available")
        add(GateCheck("significant_vs_champion", ok, paired["ci_low"] if paired else None, 0.0, d))
    return res
