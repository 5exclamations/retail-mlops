"""Model artifact integrity and safe loading.

* Models are serialised with skops (no arbitrary code execution on load) and only an explicit allow-list of
  reviewed types may be deserialised.
* A SHA-256 over the artifact files is stored as a registry tag at registration time and verified before the model
  is loaded, so a swapped/tampered artifact in the artifact store is detected (provenance: the tag lives in the
  registry database, not next to the artifact).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# Reviewed types skops reports for our pipelines (OneHot/Ordinal encoders, scalers, Ridge and HistGradientBoosting).
# TreePredictor: indexes node arrays without bounds checks, so a malicious file could crash (not execute code in)
# the process; the SHA-256 check below is the control for that risk.
TRUSTED_TYPES = [
    "functools.partial",
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
]


class IntegrityError(RuntimeError):
    pass


WEIGHTS = "model.skops"


def dir_sha256(path: str | Path) -> str:
    """Hash of the serialised estimator. Config/env files MLflow rewrites at registration are excluded."""
    return hashlib.sha256((Path(path) / WEIGHTS).read_bytes()).hexdigest()


def verify(path: str | Path, expected: str | None) -> str:
    root = Path(path)
    if not expected:
        raise IntegrityError("model version has no model_sha256 tag; refusing to load unverified artifact")
    # a swapped-in pickle must not be loadable even if model.skops is intact
    stray = [p.name for p in root.iterdir() if p.suffix in (".pkl", ".pickle", ".joblib", ".cloudpickle")]
    mlmodel = (root / "MLmodel").read_text() if (root / "MLmodel").exists() else ""
    if stray or "serialization_format: skops" not in mlmodel:
        raise IntegrityError(f"unexpected serialization (stray files {stray}); expected skops only")
    if not (root / WEIGHTS).exists():
        raise IntegrityError(f"{WEIGHTS} missing")
    actual = dir_sha256(root)
    if actual != expected:
        raise IntegrityError(f"model artifact hash mismatch (expected {expected[:12]}…, got {actual[:12]}…)")
    return actual
