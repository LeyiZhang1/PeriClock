"""PeriClock feature construction from annotated, normalized AnnData.

Algorithms are taken from the supplied analysis notebooks:
* Single-cell: Individual x cell-type fractions and mean log1p expression;
  saved VT -> SelectKBest -> saved scalers -> stable feature selection.
* Proportion: Individual x cell-type fractions.
* Non-bulk models: retain individuals with at least 20 of the 48 model cell types.
* Bulk: Individual x mean expression across ALL cells, without cell-type filtering.

This module does not normalize `.X` and does not require chronological age.
Use the preprocessing objects saved with EACH single-cell model; never refit them.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad

SINGLE_CELL_MODELS = frozenset({"PeriClock", "PeriClock-200s", "PeriClock-700s", "PeriClock-1000s"})
PROPORTION_MODEL = "PeriClock-proportion"
BULK_MODEL = "PeriClock-bulk"


def _adata_checks(adata: ad.AnnData, individual_key: str, celltype_key: str | None = None) -> None:
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an AnnData object")
    if not adata.n_obs or not adata.n_vars:
        raise ValueError("adata has no cells or genes")
    for key in (individual_key, celltype_key):
        if key is not None:
            if key not in adata.obs:
                raise KeyError(f"Missing adata.obs column: {key}")
            if adata.obs[key].isna().any():
                raise ValueError(f"adata.obs[{key!r}] contains missing values")
    if not adata.var_names.is_unique:
        raise ValueError("adata.var_names must be unique")


def _unique_names(names: Sequence[str], name: str) -> list[str]:
    if isinstance(names, str) or names is None:
        raise ValueError(f"{name} must be a nonempty list of unique strings")
    result = list(names)
    if not result or any(not isinstance(v, str) for v in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be a nonempty list of unique strings")
    return result


def _individuals(adata: ad.AnnData, individual_key: str) -> tuple[np.ndarray, pd.Index]:
    raw = adata.obs[individual_key]
    ids = raw.astype(str)
    if ids.nunique() != raw.nunique():
        raise ValueError("Distinct Individual identifiers become identical after string conversion")
    codes, donors = pd.factorize(ids, sort=True)
    return codes, pd.Index(donors, name=individual_key)


def _proportions(adata: ad.AnnData, individual_key: str, celltype_key: str,
                 cell_types: Sequence[str]) -> pd.DataFrame:
    types = _unique_names(cell_types, "cell_types")
    individuals = adata.obs[individual_key].astype(str)
    labels = adata.obs[celltype_key].astype(str)
    counts = pd.crosstab(individuals, labels)
    # Denominator includes ALL annotated cells (also types not used by the model).
    props = counts.div(counts.sum(axis=1), axis=0)
    props = props.reindex(columns=types, fill_value=0.0)
    props.index.name = individual_key
    return props


def _filter_coverage(props: pd.DataFrame, max_zeros: int | None) -> pd.DataFrame:
    """Filter non-bulk donors by coverage across the 48 trained cell types.

    The original threshold was max_zeros=28 for 48 cell types. The clarified
    intended rule is **at least 20 observed types**, so exactly 28 zeros
    (20 present types) must be retained: zero_counts <= 28.
    """
    if props.shape[1] != 48:
        raise ValueError(
            f"Non-bulk coverage requires the 48 trained cell types; got {props.shape[1]}"
        )
    if max_zeros is None:
        return props
    if not isinstance(max_zeros, (int, np.integer)) or not 0 <= max_zeros <= 48:
        raise ValueError("max_zeros must be an integer between 0 and 48 or None")
    zero_counts = (props == 0).sum(axis=1)
    return props.loc[zero_counts <= max_zeros].copy()


def _mean_expressions(adata: ad.AnnData, individual_key: str, celltype_key: str,
                      features: Sequence[str], cell_types: Sequence[str],
                      donors: pd.Index) -> pd.DataFrame:
    """Compute only requested CT_gene means; absent CT/gene => 0 as in original predictor.

    Original training constructed all genes and then filled missing columns with 0.
    Computing the required subset is algebraically identical for those columns.
    """
    requested = _unique_names(features, "expression features") if features else []
    out = pd.DataFrame(0.0, index=donors, columns=requested, dtype=np.float64)
    if not requested or len(donors) == 0:
        return out

    # A feature may contain underscores in both its cell-type and gene names.
    ordered_ct = sorted(set(cell_types), key=len, reverse=True)
    parsed: dict[str, list[tuple[str, str]]] = {}
    for f in requested:
        ct = next((ct for ct in ordered_ct if f.startswith(ct + "_")), None)
        if ct is None:
            raise ValueError(f"Cannot parse cell-type prefix from feature {f!r}")
        parsed.setdefault(ct, []).append((f, f[len(ct) + 1:]))

    gene_index = adata.var_names
    missing_genes = sorted({gene for pairs in parsed.values() for _, gene in pairs
                            if gene not in gene_index})
    if missing_genes:
        warnings.warn(f"{len(missing_genes)} expression genes absent; corresponding values are 0 "
                      f"as in the original single-cell prediction code: {missing_genes[:10]}",
                      stacklevel=2)
    donor_values = adata.obs[individual_key].astype(str).to_numpy()
    type_values = adata.obs[celltype_key].astype(str).to_numpy()
    for ct, pairs in parsed.items():
        available = [(f, gene) for f, gene in pairs if gene in gene_index]
        if not available:
            continue
        gene_positions = gene_index.get_indexer([gene for _, gene in available])
        mask_type = type_values == ct
        for donor in donors:
            rows = np.flatnonzero(mask_type & (donor_values == donor))
            if not len(rows):
                continue
            expression = adata.X[rows, :][:, gene_positions]
            mean = expression.mean(axis=0)
            result = np.asarray(mean).ravel()
            out.loc[donor, [f for f, _ in available]] = result
    return out


def build_proportion_features(adata: ad.AnnData, *, individual_key: str, celltype_key: str,
                              feature_names: Sequence[str], max_zeros: int | None = 28) -> pd.DataFrame:
    """Individual x cell-type proportions in saved model feature order."""
    _adata_checks(adata, individual_key, celltype_key)
    types = _unique_names(feature_names, "feature_names")
    result = _filter_coverage(_proportions(adata, individual_key, celltype_key, types), max_zeros)
    if result.empty:
        raise ValueError(
            "No individuals meet the non-bulk cell-type coverage requirement "
            "(at least 20 of the 48 model cell types when max_zeros=28)."
        )
    return result


def build_bulk_features(adata: ad.AnnData, *, individual_key: str,
                        feature_names: Sequence[str],
                        model_bundle: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Original donor mean; NSCLC-style missing genes use training scaler means.

    Full-gene normalization must already have been done before calling this.
    """
    _adata_checks(adata, individual_key)
    genes = _unique_names(feature_names, "feature_names")
    available = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    if not available:
        raise ValueError("No required bulk genes are present in the query")

    # Compute donor means on available genes, exactly as in the NSCLC notebook.
    indices = adata.var_names.get_indexer(available)
    expression = adata.X[:, indices]
    if sparse.issparse(expression):
        expression = expression.tocsr().astype(np.float32)
    else:
        expression = np.asarray(expression, dtype=np.float32)
    codes, donors = _individuals(adata, individual_key)
    n_cells, n_donors = len(codes), len(donors)
    counts = np.bincount(codes, minlength=n_donors)
    group = sparse.csr_matrix((np.ones(n_cells, dtype=np.float32),
                              (codes, np.arange(n_cells))), shape=(n_donors, n_cells))
    bulk = group @ expression
    if sparse.issparse(bulk):
        bulk = bulk.toarray()
    bulk = np.asarray(bulk, dtype=np.float32) / counts[:, None]
    result = pd.DataFrame(bulk, index=donors, columns=available)

    if missing:
        if model_bundle is None:
            raise ValueError(
                f"Missing {len(missing)} required bulk genes: {missing}. "
                "Pass model_bundle=load_model('PeriClock-bulk') so missing genes "
                "can be imputed with the training StandardScaler means."
            )
        if not isinstance(model_bundle, Mapping) or "model" not in model_bundle:
            raise ValueError("Bulk model_bundle must include the saved 'model' Pipeline")
        model = model_bundle["model"]
        try:
            scaler = model.named_steps["standardscaler"]
        except (AttributeError, KeyError) as exc:
            raise ValueError("Bulk model Pipeline has no saved 'standardscaler'") from exc
        if len(scaler.mean_) != len(genes):
            raise ValueError("Bulk scaler mean_ length does not match feature_names")
        if hasattr(scaler, "feature_names_in_") and list(scaler.feature_names_in_) != genes:
            raise ValueError("Bulk scaler feature_names_in_ differs from feature_names order")
        training_mean = pd.Series(scaler.mean_, index=genes)
        result = result.reindex(columns=genes).fillna(training_mean)
        warnings.warn(
            f"Imputed {len(missing)} missing bulk genes using training StandardScaler means: {missing}",
            stacklevel=2
        )
    else:
        result = result.loc[:, genes]
    return result


def _bundle(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        import joblib
        value = joblib.load(value)
    if not isinstance(value, Mapping):
        raise TypeError("single-cell model_bundle must be a saved preprocessing dictionary or its .pkl path")
    keys = ("vt", "skb", "scaler_prop", "scaler_expr", "prop_cols", "expr_cols", "selected_features")
    missing = [k for k in keys if k not in value]
    if missing:
        raise ValueError(f"single-cell model_bundle lacks original preprocessing fields: {missing}")
    return value


def build_single_cell_features(adata: ad.AnnData, *, individual_key: str, celltype_key: str,
                               model_bundle: Mapping[str, Any] | str | Path,
                               max_zeros: int | None = 28) -> pd.DataFrame:
    """Reproduce original training preprocessing and return selected *scaled* features.

    The selection/scale sequence is VT -> SelectKBest -> scaler_expr,
    and scaler_prop.transform on ratios. It is essential to use objects
    fitted during training (no fitting occurs here).
    """
    _adata_checks(adata, individual_key, celltype_key)
    bundle = _bundle(model_bundle)
    prop_cols = _unique_names(bundle["prop_cols"], "prop_cols")
    expr_cols = _unique_names(bundle["expr_cols"], "expr_cols")
    selected = _unique_names(bundle["selected_features"], "selected_features")
    if any(not f.endswith("_ratio") for f in prop_cols):
        raise ValueError("Saved prop_cols must have the training suffix '_ratio'")
    cell_types = [p[:-6] for p in prop_cols]
    props = _filter_coverage(_proportions(adata, individual_key, celltype_key, cell_types), max_zeros)
    donors = props.index
    if len(donors) == 0:
        raise ValueError(
            "No individuals meet the non-bulk cell-type coverage requirement "
            "(at least 20 of the 48 model cell types when max_zeros=28)."
        )

    # Saved VT/SKB support identifies exactly the 8,000-ish expression features
    # used to fit scaler_expr. No need to materialize all >1M candidate columns.
    vt_mask = np.asarray(bundle["vt"].get_support(), dtype=bool)
    if len(vt_mask) != len(expr_cols):
        raise ValueError("Saved VarianceThreshold mask length differs from expr_cols")
    vt_names = np.asarray(expr_cols, dtype=object)[vt_mask]
    skb_mask = np.asarray(bundle["skb"].get_support(), dtype=bool)
    if len(skb_mask) != len(vt_names):
        raise ValueError("Saved SelectKBest mask length differs from VT-selected names")
    selected_expr_names = vt_names[skb_mask].tolist()
    all_names = prop_cols + selected_expr_names
    if not set(selected).issubset(all_names):
        raise ValueError("selected_features missing from saved VT/SKB feature definitions")

    expr = _mean_expressions(adata, individual_key, celltype_key,
                             selected_expr_names, cell_types, donors)
    prop_raw = props.to_numpy(dtype=np.float64)
    expr_raw = expr.to_numpy(dtype=np.float64)
    prop_scaled = bundle["scaler_prop"].transform(prop_raw)
    expr_scaled = bundle["scaler_expr"].transform(expr_raw)
    full = np.hstack([prop_scaled, expr_scaled])
    all_scaled = pd.DataFrame(full, index=donors, columns=all_names)
    return all_scaled.loc[:, selected].copy()


def build_features(adata: ad.AnnData, *, model_name: str,
                   individual_key: str, celltype_key: str | None = None,
                   feature_names: Sequence[str] | None = None,
                   model_bundle: Mapping[str, Any] | str | Path | None = None,
                   max_zeros: int | None = 28) -> pd.DataFrame:
    """Construct an Individual x model-input-feature matrix.

    single_cell: provide the original saved preprocessing bundle (per model).
    proportion/bulk: provide their saved model feature_names.
    No `Age` is needed; the returned columns are prediction inputs only.
    """
    if model_name == BULK_MODEL:
        if feature_names is None:
            raise ValueError("Bulk requires model feature_names")
        return build_bulk_features(adata, individual_key=individual_key, feature_names=feature_names,
                                   model_bundle=model_bundle)
    if celltype_key is None:
        raise ValueError(f"{model_name} requires celltype_key")
    if model_name == PROPORTION_MODEL:
        if feature_names is None:
            raise ValueError("Proportion requires model feature_names")
        return build_proportion_features(adata, individual_key=individual_key,
                                         celltype_key=celltype_key, feature_names=feature_names,
                                         max_zeros=max_zeros)
    if model_name in SINGLE_CELL_MODELS:
        if model_bundle is None:
            raise ValueError("Single-cell models require their original saved preprocessing bundle "
                             "(vt, skb, scaler_prop, scaler_expr, prop_cols, expr_cols, selected_features)")
        return build_single_cell_features(adata, individual_key=individual_key,
                                          celltype_key=celltype_key,
                                          model_bundle=model_bundle, max_zeros=max_zeros)
    raise ValueError(f"Unsupported model_name: {model_name!r}")
