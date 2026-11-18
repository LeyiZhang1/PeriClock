import numpy as np
from scipy import sparse
from ._optional import require_dependency

sc = require_dependency("scanpy", "preprocessing")


# ============================================================
# 1. Normalize and log1p
# ============================================================
def normalize_expression(
    adata,
    target_sum=1e4,
    copy=True
):
    """
    Normalize raw UMI counts and apply log1p.

    Parameters
    ----------
    adata : AnnData
        Raw UMI count matrix stored in adata.X.

    target_sum : float
        Target library size per cell. Default: 10000.

    copy : bool
        Whether to create a copy.

    Returns
    -------
    AnnData
        Normalized and log1p-transformed AnnData.
    """

    if copy:
        adata = adata.copy()

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("AnnData is empty.")

    X = adata.X

    # Check numerical validity
    values = X.data if sparse.issparse(X) else np.asarray(X)

    if not np.isfinite(values).all():
        raise ValueError(
            "Expression matrix contains NaN or infinite values."
        )

    if np.any(values < 0):
        raise ValueError(
            "Raw count matrix contains negative values."
        )

    # Normalize total counts per cell
    sc.pp.normalize_total(
        adata,
        target_sum=target_sum
    )

    # Log transformation
    sc.pp.log1p(adata)

    adata.uns["periclock_normalization"] = {
        "method": "normalize_total_log1p",
        "target_sum": target_sum
    }

    return adata
