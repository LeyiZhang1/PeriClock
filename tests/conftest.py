"""Local source tests; no package installation or model fitting is performed."""
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from periclock import list_models, load_model


@pytest.fixture(scope="session")
def release_root():
    return ROOT


def pytest_collection_modifyitems(items):
    # Pytest includes transitive fixture dependencies in fixturenames.
    for item in items:
        if "real_model_dir" in item.fixturenames:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def real_model_dir():
    configured = os.environ.get("PERICLOCK_MODEL_DIR")
    if not configured:
        pytest.skip("Real models not configured; set PERICLOCK_MODEL_DIR to the downloaded model directory")
    directory = Path(configured).expanduser().resolve()
    # An explicitly configured incomplete release is an error, never a skip.
    from periclock.loader import MODEL_FILES
    missing = [filename for filename in MODEL_FILES.values()
               if not (directory / filename).is_file()]
    if missing:
        pytest.fail(f"Incomplete PERICLOCK_MODEL_DIR={directory}: missing {', '.join(missing)}")
    return directory


@pytest.fixture(scope="session")
def bundles(real_model_dir):
    return {name: load_model(name, model_dir=real_model_dir) for name in list_models()}


@pytest.fixture
def make_adata():
    def make(values, donors, labels=None, genes=None, csr=False):
        values = np.asarray(values, dtype=np.float64)
        obs = pd.DataFrame({"donor": donors}, index=[f"cell{i}" for i in range(len(donors))])
        if labels is not None:
            obs["ct"] = labels
        if genes is None:
            genes = [f"G{i}" for i in range(values.shape[1])]
        return ad.AnnData(sparse.csr_matrix(values) if csr else values.copy(),
                          obs=obs, var=pd.DataFrame(index=genes))
    return make


@pytest.fixture
def synthetic_query(bundles, make_adata):
    """Small, deterministic expression values; NOT historical patient data."""
    types = [c[:-6] for c in bundles["PeriClock"]["artifact"]["prop_cols"]]
    labels = (types + ["unknown", types[0]]) * 2
    values = (np.arange(len(labels) * 3).reshape(-1, 3) % 13 + 1) / 3
    return make_adata(values, ["donor_b"] * 50 + ["donor_a"] * 50, labels,
                      ["NAP1L2", "GTSCR1", "MT-CO1"], csr=True)
