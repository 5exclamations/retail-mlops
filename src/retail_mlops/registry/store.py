"""Model registry operations on top of MLflow aliases: candidate -> champion, previous (for rollback)."""
from __future__ import annotations

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from retail_mlops.config import Config


def client_for(cfg: Config) -> MlflowClient:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    return MlflowClient()


def alias_version(client: MlflowClient, cfg: Config, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(cfg.mlflow.model_name, alias).version)
    except MlflowException:
        return None


def register_version(client: MlflowClient, cfg: Config, run_id: str, tags: dict[str, str]) -> str:
    mv = mlflow.register_model(f"runs:/{run_id}/model", cfg.mlflow.model_name, tags=tags)
    return str(mv.version)


def promote(client: MlflowClient, cfg: Config, version: str) -> None:
    name, a = cfg.mlflow.model_name, cfg.mlflow
    current = alias_version(client, cfg, a.champion_alias)
    if current and current != str(version):
        client.set_registered_model_alias(name, a.previous_alias, current)
    client.set_registered_model_alias(name, a.champion_alias, str(version))
    client.set_model_version_tag(name, str(version), "stage", "champion")
    if current and current != str(version):
        client.set_model_version_tag(name, current, "stage", "previous")


def rollback(client: MlflowClient, cfg: Config) -> str:
    """Point 'champion' back at the 'previous' version. The API picks it up on its next reload check."""
    a, name = cfg.mlflow, cfg.mlflow.model_name
    prev = alias_version(client, cfg, a.previous_alias)
    if prev is None:
        raise RuntimeError("no previous version to roll back to")
    current = alias_version(client, cfg, a.champion_alias)
    client.set_registered_model_alias(name, a.champion_alias, prev)
    client.set_model_version_tag(name, prev, "stage", "champion")
    if current:
        client.set_registered_model_alias(name, a.previous_alias, current)  # allows rolling forward again
        client.set_model_version_tag(name, current, "stage", "rolled_back")
    return prev
