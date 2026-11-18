"""Reference-based cell-type annotation for PeriClock.

All annotation functions expect log1p(CP10K) expression in `.X` unless the
user's reference/model was explicitly trained with another preprocessing.
Optional integrations are imported lazily.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import warnings
import logging

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from ._optional import require_dependency

sc = require_dependency("scanpy", "annotation")
logger = logging.getLogger(__name__)

AnnInput = Union[ad.AnnData, str, Path]

# Legacy fallback only; the public code release does not bundle this asset.
DEFAULT_CELLTYPIST_MODEL = (
    Path(__file__).resolve().parent
    / "celltypist_model"
    / "celltypist_model_cell_subtype_final.pkl"
)


def _load(value: AnnInput) -> ad.AnnData:
    if isinstance(value, (str, Path)):
        path = _resource_path(value)
        return sc.read_h5ad(str(path))
    return value


def _resource_path(value) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Annotation data not found: {path}. Reference and query data are "
            "downloaded separately; pass their actual .h5ad paths."
        )
    return path


def _validate(query: ad.AnnData) -> None:
    if query.n_obs == 0 or query.n_vars == 0:
        raise ValueError('Query cannot be empty.')
    if not query.obs_names.is_unique or not query.var_names.is_unique:
        raise ValueError('Query cell and gene identifiers must be unique.')


def _output(query: ad.AnnData, labels, scores, *, method: str,
            label_key: str, copy: bool = True, score_name: str = 'annotation_confidence') -> ad.AnnData:
    out = query.copy() if copy else query
    labels = pd.Series(np.asarray(labels, dtype=object), index=query.obs_names)
    scores = pd.Series(np.asarray(scores, dtype=float), index=query.obs_names)
    if labels.isna().any() or scores.isna().any():
        raise ValueError('Missing predictions or scores; annotation was not written.')
    out.obs[label_key] = labels.reindex(out.obs_names).to_numpy()
    out.obs[score_name] = scores.reindex(out.obs_names).to_numpy()
    out.uns['annotation_method'] = method
    out.uns['annotation_score_type'] = (
        'weighted BBKNN reference-neighbor vote share' if method == 'bbknn_knn' else
        'CellTypist sigmoid score for predicted class (not calibrated multiclass probability)' if method == 'celltypist' else
        'distance-weighted KNN class support in Symphony space'
    )
    return out


def annotate_celltypist(query: AnnInput, *, model_path: Optional[Union[str, Path]] = None,
                       label_key: str = 'cell_subtype_final',
                       confidence_threshold: Optional[float] = None,
                       min_gene_overlap: float = 0.0,
                       copy: bool = True) -> ad.AnnData:
    """Annotate log1p(CP10K) query with a pretrained CellTypist model.

    Pass model_path to the separately downloaded compatible CellTypist model.
    The package-local default is retained only for legacy checkouts.
    """
    celltypist = require_dependency("celltypist", "celltypist")
    from celltypist import models
    query = _load(query)
    _validate(query)
    if not 0 <= min_gene_overlap <= 1:
        raise ValueError('min_gene_overlap must be in [0, 1].')
    resolved_model_path = Path(model_path).expanduser() if model_path is not None else DEFAULT_CELLTYPIST_MODEL
    if not resolved_model_path.is_file():
        raise FileNotFoundError(
            f'CellTypist model not found: {resolved_model_path}. Download the '
            'compatible annotation model separately and pass model_path="/path/to/model.pkl".'
        )
    model = models.Model.load(model=str(resolved_model_path))
    common = query.var_names.intersection(pd.Index(model.features))
    frac = len(common) / len(model.features)
    if frac < min_gene_overlap or not len(common):
        raise ValueError(f'Only {len(common)}/{len(model.features)} model genes overlap query ({frac:.1%}).')
    if frac < 0.5:
        warnings.warn(f'Low CellTypist model gene overlap: {frac:.1%}.', stacklevel=2)
    result = celltypist.annotate(query, model=model, majority_voting=False)
    labels = result.predicted_labels.loc[query.obs_names, 'predicted_labels'].astype(str)
    probs = result.probability_matrix.loc[query.obs_names]
    if not probs.columns.is_unique:
        raise ValueError('Duplicated CellTypist probability columns.')
    if not labels.isin(probs.columns).all():
        raise ValueError('Predicted CellTypist class is absent from probability matrix.')
    scores = probs.to_numpy()[np.arange(query.n_obs), probs.columns.get_indexer(labels)]
    if confidence_threshold is not None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError('confidence_threshold must be in [0, 1].')
        labels = labels.mask(scores < confidence_threshold, 'unknown')
    out = _output(query, labels, scores, method='celltypist', label_key=label_key, copy=copy)
    out.uns['annotation_gene_overlap'] = {'matched': len(common), 'model_total': len(model.features), 'fraction': frac}
    return out


def _stratified_reference(ref: ad.AnnData, label_key: str, max_per_type: Optional[int], seed: int) -> ad.AnnData:
    if label_key not in ref.obs:
        raise KeyError(f'Reference lacks {label_key!r}.')
    indices = np.flatnonzero(ref.obs[label_key].notna().to_numpy())
    if not len(indices):
        raise ValueError('Reference has no labeled cells.')
    if max_per_type is None:
        return ref[indices, :].copy()
    if max_per_type < 1:
        raise ValueError('max_per_type must be >= 1.')
    rng = np.random.default_rng(seed)
    selected = []
    valid_labels = ref.obs[label_key].iloc[indices].astype(str)
    for _, positions in valid_labels.groupby(valid_labels, sort=False).indices.items():
        group_idx = indices[np.asarray(positions)]
        selected.extend(rng.choice(group_idx, size=min(len(group_idx), max_per_type), replace=False))
    return ref[np.sort(np.asarray(selected, dtype=int)), :].copy()


def annotate_bbknn_knn(query: AnnInput, *, reference: AnnInput,
                        reference_label_key: str = 'cell_subtype_final',
                        label_key: str = 'cell_subtype_final',
                        max_per_type: Optional[int] = 4000,
                        n_hvg: int = 2000, n_pcs: int = 50,
                        neighbors_within_batch: int = 10,
                        k: int = 20, confidence_threshold: float = 0.5,
                        random_state: int = 0, copy: bool = True) -> ad.AnnData:
    """Joint PCA + BBKNN, voting ONLY among labeled reference neighbors.

    The reference and query X matrices must share the same normalization.
    """
    require_dependency("bbknn", "bbknn")
    query = _load(query)
    ref = _load(reference)
    _validate(query)
    _validate(ref)
    if k < 1 or neighbors_within_batch < 1 or n_pcs < 2 or n_hvg < 2:
        raise ValueError('k, neighbors_within_batch, n_pcs, n_hvg must be positive (PCA/HVG >= 2).')
    if not 0 <= confidence_threshold <= 1:
        raise ValueError('confidence_threshold must be in [0,1].')
    ref = _stratified_reference(ref, reference_label_key, max_per_type, random_state)
    common = ref.var_names.intersection(query.var_names, sort=False)
    if len(common) < 3:
        raise ValueError('Too few shared genes for joint PCA.')
    ref_part = ref[:, common].copy()
    query_part = query[:, common].copy()
    # Use disjoint internal obs IDs to handle identical cell barcodes across datasets.
    ref_part.obs_names = pd.Index([f'ref_{i}' for i in range(ref_part.n_obs)])
    query_part.obs_names = pd.Index([f'query_{i}' for i in range(query_part.n_obs)])
    merged = ad.concat([ref_part, query_part], label='_annotation_dataset', keys=['ref', 'query'], join='inner')
    sc.pp.highly_variable_genes(merged, flavor='seurat', batch_key='_annotation_dataset',
                                n_top_genes=min(n_hvg, merged.n_vars))
    hvgs = merged.var['highly_variable'].to_numpy()
    if hvgs.sum() < 3:
        raise ValueError('Too few highly variable genes for PCA.')
    merged = merged[:, hvgs].copy()
    max_pcs = min(merged.n_obs - 1, merged.n_vars - 1)
    pcs = min(n_pcs, max_pcs)
    if pcs < 2:
        raise ValueError('Insufficient observations/features for PCA.')
    sc.tl.pca(merged, n_comps=pcs, svd_solver='arpack', random_state=random_state)
    # The BBKNN implementation is optional. It produces a balanced graph.
    sc.external.pp.bbknn(merged, batch_key='_annotation_dataset',
                         neighbors_within_batch=neighbors_within_batch, n_pcs=pcs)
    conn = merged.obsp['connectivities'].tocsr()
    n_ref = ref_part.n_obs
    ref_labels = ref.obs[reference_label_key].astype(str).to_numpy()
    labels, conf = [], []
    for i in range(n_ref, merged.n_obs):
        row = conn.getrow(i)
        is_ref = row.indices < n_ref
        idx, weights = row.indices[is_ref], row.data[is_ref]
        if not len(idx):
            labels.append('unknown'); conf.append(0.0); continue
        order = np.argsort(weights)[::-1][:k]
        idx, weights = idx[order], weights[order]
        sums = pd.Series(weights).groupby(ref_labels[idx]).sum()
        total = sums.sum()
        if not np.isfinite(total) or total <= 0:
            labels.append('unknown'); conf.append(0.0); continue
        winner = sums.idxmax()
        score = float(sums.max() / total)
        labels.append(winner if score >= confidence_threshold else 'unknown')
        conf.append(score)
    out = _output(query, labels, conf, method='bbknn_knn', label_key=label_key, copy=copy)
    out.uns['annotation_gene_overlap'] = {'shared': len(common), 'hvg': int(hvgs.sum())}
    return out


def build_symphony_reference(reference: AnnInput, *, query: AnnInput,
                             reference_label_key: str = 'cell_subtype_final',
                             exclude_batches: Optional[list[str]] = None,
                             max_per_type: int = 1000,
                             n_hvg: int = 2500, n_pcs: int = 30,
                             random_state: int = 42) -> ad.AnnData:
    """Build the Symphony reference using the ORIGINAL query-dependent pipeline.

    Original processing order is retained exactly:
    exclude reference batches -> batch-weighted sampling per cell subtype ->
    intersect reference/query genes -> remove MT- genes -> Seurat HVGs ->
    scale -> PCA -> Harmony.

    IMPORTANT: Because common genes are selected BEFORE HVGs, this reference
    depends on the query supplied here. It cannot be treated as a universal,
    query-independent prebuilt reference.

    If `reference` is an .h5ad path, reference cells are sampled in backed
    mode before their expression is loaded, as in the original script.
    """
    sy = require_dependency("symphonypy", "symphony")

    query = _load(query)
    _validate(query)
    if max_per_type < 1:
        raise ValueError('max_per_type must be >= 1.')
    if exclude_batches is None:
        exclude_batches = ['GSM6507608_1', 'GSM6507609_1', 'GSM6507610_1']

    opened_backed = isinstance(reference, (str, Path))
    ref_source = sc.read_h5ad(str(_resource_path(reference)), backed='r') if opened_backed else reference
    try:
        _validate(ref_source)
        if reference_label_key not in ref_source.obs.columns:
            raise KeyError(f'Reference lacks {reference_label_key!r}.')
        if 'batch' not in ref_source.obs.columns:
            raise KeyError("Reference lacks 'batch' needed for original exclusion and sampling.")
        batch_col = 'Study' if 'Study' in ref_source.obs.columns else 'batch'

        # Same metadata filtering and weighted sampling as the user's script.
        meta = ref_source.obs[[reference_label_key, 'batch']].copy()
        meta['_row'] = np.arange(len(meta))
        mask = (meta[reference_label_key].notna() &
                ~meta['batch'].astype(str).isin(exclude_batches))
        meta = meta.loc[mask].copy()
        rng = np.random.default_rng(random_state)
        selected_idx = []
        for ct, group in meta.groupby(reference_label_key, observed=True):
            n = min(len(group), max_per_type)
            batch_counts = group['batch'].value_counts()
            weights = 1 / np.sqrt(group['batch'].map(batch_counts).to_numpy())
            chosen = rng.choice(group['_row'].to_numpy(), size=n,
                                replace=False, p=weights / weights.sum())
            selected_idx.extend(chosen)
        selected_idx = np.sort(np.asarray(selected_idx, dtype=int))
        if len(selected_idx) == 0:
            raise ValueError('No labeled reference cells left after filtering.')
        ref = (ref_source[selected_idx, :].to_memory() if opened_backed
               else ref_source[selected_idx, :].copy())
    finally:
        if opened_backed:
            ref_source.file.close()

    ref.obs['symphony_batch'] = (
        ref.obs[batch_col].astype('object').fillna('Unknown').astype(str)
    )
    if not ref.var_names.is_unique or not query.var_names.is_unique:
        raise ValueError('Duplicated gene names found. Resolve them before mapping.')

    # Original: query-dependent gene intersection BEFORE HVG selection.
    common_genes = ref.var_names.intersection(query.var_names, sort=False)
    logger.info('Common genes: %s', len(common_genes))
    if len(common_genes) < 2000:
        raise ValueError('Too few common genes. Check gene identifiers.')
    ref = ref[:, common_genes].copy()
    mt_mask = ref.var_names.str.upper().str.startswith('MT-')
    ref = ref[:, ~mt_mask].copy()

    # Keep the original preprocessing, parameters and operation order.
    sc.pp.highly_variable_genes(ref, n_top_genes=n_hvg, flavor='seurat')
    ref = ref[:, ref.var['highly_variable']].copy()
    sc.pp.scale(ref, max_value=10)
    sc.pp.pca(ref, n_comps=n_pcs, svd_solver='arpack')
    sy.pp.harmony_integrate(
        ref, key='symphony_batch', random_seed=random_state,
        nclust=50, max_iter_harmony=20, verbose=True
    )
    return ref


def annotate_symphony(query: AnnInput, *, reference: AnnInput,
                      reference_label_key: str = 'cell_subtype_final',
                      label_key: str = 'cell_subtype_final',
                      k: int = 15, copy: bool = True,
                      plot: bool = False) -> ad.AnnData:
    """Original Symphony mapping and distance-weighted KNN label transfer.

    `reference` must be built using `build_symphony_reference(..., query=query)`
    for the SAME query to reproduce the original query-dependent pipeline.
    Labels/support are also written into the unified PeriClock output columns.
    """
    sy = require_dependency("symphonypy", "symphony")

    query = _load(query)
    reference = _load(reference)
    _validate(query)
    _validate(reference)
    if reference_label_key not in reference.obs:
        raise KeyError(f'Reference lacks {reference_label_key!r}.')

    work = query.copy() if copy else query
    sy.tl.map_embedding(adata_query=work, adata_ref=reference,
                        key=None, use_genes_column=None)
    logger.info('Mapped Query: %s', work.obsm['X_pca_harmony'].shape)

    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
    knn.fit(reference.obsm['X_pca_harmony'], reference.obs[reference_label_key].astype(str))
    pred_proba = knn.predict_proba(work.obsm['X_pca_harmony'])
    labels = knn.classes_[pred_proba.argmax(axis=1)]
    support = pred_proba.max(axis=1)

    # Original result column names; PeriClock-compatible aliases are additional.
    work.obs['symphony_pred'] = pd.Categorical(labels)
    work.obs['symphony_knn_support'] = support
    work.obs[label_key] = labels
    work.obs['annotation_confidence'] = support
    work.uns['annotation_method'] = 'symphony'
    work.uns['annotation_score_type'] = 'distance-weighted KNN class support in Symphony space'

    if plot:
        import matplotlib.pyplot as plt
        sc.pp.neighbors(reference, use_rep='X_pca_harmony', n_neighbors=15)
        sc.tl.umap(reference, random_state=42)
        if 'X_umap' in work.obsm:
            work.obsm['X_umap_original'] = work.obsm['X_umap'].copy()
        sy.tl.ingest(adata_query=work, adata_ref=reference, use_rep='X_pca_harmony')
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(reference.obsm['X_umap'][:, 0], reference.obsm['X_umap'][:, 1],
                   s=1, c='lightgrey', alpha=0.4, label='Healthy reference', rasterized=True)
        ax.scatter(work.obsm['X_umap'][:, 0], work.obsm['X_umap'][:, 1],
                   s=2, c='#D55E00', alpha=0.7, label='Query', rasterized=True)
        ax.set(xlabel='UMAP1', ylabel='UMAP2', title='Symphony reference mapping')
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()
        sc.pl.umap(work, color=['symphony_pred', 'symphony_knn_support'],
                   legend_loc='none', frameon=False, wspace=0.35)
        if 'leiden' in work.obs:
            cluster_ct = pd.crosstab(work.obs['leiden'], work.obs['symphony_pred'],
                                    normalize='index')
            print('\n===== Leiden × Symphony =====')
            print(cluster_ct.round(3).to_string())
            for cluster in cluster_ct.index:
                print(f'\nCluster {cluster}')
                print((cluster_ct.loc[cluster].nlargest(3) * 100).round(2).to_string())

    return work


def annotate_symphony_original(query: AnnInput, *, reference: AnnInput,
                               reference_label_key: str = 'cell_subtype_final',
                               exclude_batches: Optional[list[str]] = None,
                               max_per_type: int = 1000,
                               n_hvg: int = 2500, n_pcs: int = 30,
                               random_state: int = 42,
                               k: int = 15, label_key: str = 'cell_subtype_final',
                               copy: bool = True, plot: bool = False) -> ad.AnnData:
    """Run the full original query-dependent Symphony pipeline end to end."""
    ref = build_symphony_reference(
        reference, query=query, reference_label_key=reference_label_key,
        exclude_batches=exclude_batches, max_per_type=max_per_type,
        n_hvg=n_hvg, n_pcs=n_pcs, random_state=random_state
    )
    return annotate_symphony(
        query, reference=ref, reference_label_key=reference_label_key,
        label_key=label_key, k=k, copy=copy, plot=plot
    )


def annotate_cells(query: AnnInput, *, method: str, **kwargs) -> ad.AnnData:
    """Dispatch annotation methods. Symphony runs the complete original pipeline."""
    methods = {'bbknn_knn': annotate_bbknn_knn,
               'celltypist': annotate_celltypist,
               'symphony': annotate_symphony_original}
    if method not in methods:
        raise ValueError(f'Unsupported annotation method {method!r}. Choose from {tuple(methods)}.')
    return methods[method](query, **kwargs)
