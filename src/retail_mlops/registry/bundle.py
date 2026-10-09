"""A ModelBundle is everything needed to serve a registered model identically to how it was trained:
the fitted sklearn pipeline, the exact config it was trained with, and its monitoring reference data."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from retail_mlops.config import Config
from retail_mlops.features.engineering import build_features, feature_columns

BUNDLE_DIR = "bundle"


@dataclass
class ModelBundle:
    model: object
    cfg: Config  # config the model was trained with (NOT the current repo config)
    name: str
    version: str
    run_id: str
    metadata: dict = field(default_factory=dict)
    reference: pd.DataFrame | None = None

    @property
    def columns(self) -> list[str]:
        return feature_columns(self.cfg.forecast)

    @property
    def reference_wape(self) -> float | None:
        return self.metadata.get("metrics", {}).get("test", {}).get("wape")

    def predict_features(self, feats: pd.DataFrame):
        return self.model.predict(feats[self.columns]).clip(min=0.0)

    def score_panel(self, panel: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
        """Score the rows identified by keys (store_id, sku_id, date) of a gap-free panel with this model's own
        feature recipe. Used to evaluate the current champion on a new holdout."""
        feats = build_features(panel, self.cfg.forecast)
        feats = feats.merge(keys[["store_id", "sku_id", "date"]], on=["store_id", "sku_id", "date"], how="inner")
        feats["prediction"] = self.predict_features(feats)
        return feats


def log_bundle_files(cfg: Config, metadata: dict, reference: pd.DataFrame) -> None:
    """Log config / metadata / reference into the active MLflow run under bundle/."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(cfg.model_dump_json(indent=2))
        (Path(d) / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        reference.to_parquet(Path(d) / "reference.parquet", index=False)
        mlflow.log_artifacts(d, artifact_path=BUNDLE_DIR)


def load_bundle(cfg: Config, version: str | None = None, alias: str | None = None) -> ModelBundle:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient()
    name = cfg.mlflow.model_name
    mv = (client.get_model_version(name, str(version)) if version is not None
          else client.get_model_version_by_alias(name, alias or cfg.mlflow.champion_alias))
    model = mlflow.sklearn.load_model(f"models:/{name}/{mv.version}")
    with tempfile.TemporaryDirectory() as d:
        local = Path(client.download_artifacts(mv.run_id, BUNDLE_DIR, d))
        trained_cfg = Config.model_validate_json((local / "config.json").read_text())
        metadata = json.loads((local / "metadata.json").read_text())
        reference = pd.read_parquet(local / "reference.parquet")
    # the serving process decides where data lives; the model decides how features are built
    return ModelBundle(model, trained_cfg, name, str(mv.version), mv.run_id, metadata, reference)
