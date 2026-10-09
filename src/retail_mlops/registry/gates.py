"""Automated promotion gates. Pure functions, so they are trivially unit-testable."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

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

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}

    def summary(self) -> str:
        return "\n".join(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks)


def evaluate_gates(
    cfg: GatesCfg,
    candidate: dict[str, float],
    best_baseline: dict[str, float],
    candidate_segments: dict[str, dict[str, float]],
    baseline_segments: dict[str, dict[str, float]],
    champion: dict[str, float] | None = None,
    data_valid: bool = True,
) -> GateResult:
    """All metric dicts are computed on the SAME held-out test window (the champion is re-scored on it)."""
    res = GateResult()
    add = res.checks.append

    if cfg.require_valid_data:
        add(GateCheck("data_validation", data_valid, None, None, "raw data passed validation" if data_valid else "raw data FAILED validation"))

    cw, bw = candidate["wape"], best_baseline["wape"]
    improvement = (bw - cw) / bw if bw > 0 else 0.0
    add(GateCheck("beats_baseline", improvement >= cfg.min_improvement_vs_baseline, improvement,
                  cfg.min_improvement_vs_baseline,
                  f"candidate WAPE {cw:.4f} vs best baseline {bw:.4f} ({improvement:+.1%} relative, need >= {cfg.min_improvement_vs_baseline:.1%})"))

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
    ok = worst <= cfg.max_segment_regression_vs_baseline
    add(GateCheck("no_segment_regression", ok, worst if worst_seg else None, cfg.max_segment_regression_vs_baseline,
                  f"worst category vs baseline: {worst_seg or 'n/a'} ({worst:+.1%}, max {cfg.max_segment_regression_vs_baseline:+.1%})"))

    if champion is not None:
        rel = (cw - champion["wape"]) / champion["wape"]
        add(GateCheck("not_worse_than_champion", rel <= cfg.max_regression_vs_champion, rel, cfg.max_regression_vs_champion,
                      f"candidate WAPE {cw:.4f} vs champion {champion['wape']:.4f} on the same holdout ({rel:+.2%}, max {cfg.max_regression_vs_champion:+.2%})"))
    else:
        add(GateCheck("not_worse_than_champion", True, None, None, "no champion yet; first model only needs to clear the other gates"))
    return res
