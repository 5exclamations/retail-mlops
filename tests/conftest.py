import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from retail_mlops.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from retail_mlops.data.generate import generate_sales  # noqa: E402

SMOKE = DEFAULT_CONFIG_PATH.parent / "smoke.yaml"


@pytest.fixture(scope="session")
def smoke_cfg():
    return load_config(DEFAULT_CONFIG_PATH, SMOKE)


@pytest.fixture(scope="session")
def raw(smoke_cfg):
    return generate_sales(smoke_cfg)


@pytest.fixture()
def cfg(smoke_cfg, tmp_path):
    """Per-test config with isolated MLflow store and database."""
    c = smoke_cfg.model_copy(deep=True)
    c.mlflow.tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    c.database_url = os.getenv("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/retail.db"
    if os.getenv("TEST_DATABASE_URL"):  # shared server: start every test from empty tables
        from retail_mlops.db import get_engine, metadata

        eng = get_engine(c.database_url)
        metadata.drop_all(eng)
        eng.dispose()
    return c
