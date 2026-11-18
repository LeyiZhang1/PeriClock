import numpy as np
import logging
from ._optional import require_dependency

sc = require_dependency("scanpy", "preprocessing")
logger = logging.getLogger(__name__)


def calculate_qc_metrics(
    adata,
    mitochondrial_prefix="MT-"
):
    """
    Calculate single-cell QC metrics.

    Parameters
    ----------
    adata : AnnData
        Raw count matrix.
    mitochondrial_prefix : str
        Prefix used to identify mitochondrial genes.

    Returns
    -------
    AnnData
        Copy of AnnData with QC metrics in .obs.
    """
    adata = adata.copy()

    adata.var["mt"] = adata.var_names.str.startswith(
        mitochondrial_prefix
    )

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        inplace=True
    )

    return adata


def filter_cells(
    adata,
    min_genes=None,
    max_genes=None,
    min_counts=None,
    max_counts=None,
    max_pct_mt=None
):
    """
    Filter cells using explicitly specified QC thresholds.

    No filtering is applied when all thresholds are None.
    """
    adata = adata.copy()

    required_columns = {
        "min_genes": "n_genes_by_counts",
        "max_genes": "n_genes_by_counts",
        "min_counts": "total_counts",
        "max_counts": "total_counts",
        "max_pct_mt": "pct_counts_mt"
    }

    thresholds = {
        "min_genes": min_genes,
        "max_genes": max_genes,
        "min_counts": min_counts,
        "max_counts": max_counts,
        "max_pct_mt": max_pct_mt
    }

    for threshold, value in thresholds.items():
        if value is not None:
            column = required_columns[threshold]

            if column not in adata.obs:
                raise ValueError(
                    f"QC metric '{column}' is missing. "
                    "Run calculate_qc_metrics() first."
                )

    mask = np.ones(adata.n_obs, dtype=bool)

    if min_genes is not None:
        mask &= adata.obs["n_genes_by_counts"].to_numpy() >= min_genes

    if max_genes is not None:
        mask &= adata.obs["n_genes_by_counts"].to_numpy() <= max_genes

    if min_counts is not None:
        mask &= adata.obs["total_counts"].to_numpy() >= min_counts

    if max_counts is not None:
        mask &= adata.obs["total_counts"].to_numpy() <= max_counts

    if max_pct_mt is not None:
        mask &= adata.obs["pct_counts_mt"].to_numpy() <= max_pct_mt

    n_before = adata.n_obs
    adata = adata[mask].copy()

    logger.info("Cells before QC: %s", n_before)
    logger.info("Cells after QC: %s", adata.n_obs)
    logger.info("Cells removed: %s", n_before - adata.n_obs)

    if adata.n_obs == 0:
        raise ValueError("All cells were removed by QC filtering.")

    return adata


def run_qc(adata, **kwargs):
    """
    Calculate QC metrics and optionally filter cells.
    """
    adata = calculate_qc_metrics(adata)
    adata = filter_cells(adata, **kwargs)
    return adata
