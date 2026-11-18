import numpy as np
import pandas as pd


# ============================================================
# 1. Validate AnnData
# ============================================================
def validate_anndata(
    adata,
    donor_col="Individual",
    celltype_col="cell_subtype_final"
):
    """Check AnnData structure and required metadata."""

    if adata.n_obs == 0:
        raise ValueError("AnnData contains no cells.")

    if adata.n_vars == 0:
        raise ValueError("AnnData contains no genes.")

    for col in [donor_col, celltype_col]:
        if col not in adata.obs.columns:
            raise ValueError(
                f"Required column '{col}' not found in adata.obs."
            )

        if adata.obs[col].isna().any():
            raise ValueError(
                f"Column '{col}' contains missing values."
            )

    if not adata.var_names.is_unique:
        raise ValueError("Duplicate gene names found.")

    if not adata.obs_names.is_unique:
        raise ValueError("Duplicate cell IDs found.")

    return True


# ============================================================
# 2. Check missing features
# ============================================================
def check_missing_features(df, required_features):
    """Return model features absent from a DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if not df.columns.is_unique:
        raise ValueError("Duplicate feature names found.")

    return [
        feature
        for feature in required_features
        if feature not in df.columns
    ]


# ============================================================
# 3. Validate numerical matrix
# ============================================================
def validate_numeric_matrix(df):
    """Check whether a feature matrix contains valid numbers."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    try:
        values = df.to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Feature matrix contains non-numeric values."
        ) from exc

    if not np.isfinite(values).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    return True