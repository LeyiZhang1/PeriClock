"""Analytic checks plus characterization of unresolved input-state behavior."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from periclock.normalization import normalize_expression
from periclock.qc import calculate_qc_metrics, filter_cells, run_qc


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


@pytest.mark.parametrize("csr", [False, True])
def test_qc_metrics_and_filtering(make_adata, csr):
    a = make_adata([[1, 3, 0], [0, 2, 2], [8, 0, 0]], ["a", "a", "b"],
                   genes=["MT-CO1", "G1", "G2"], csr=csr)
    before = dense(a.X).copy()
    q = calculate_qc_metrics(a)
    np.testing.assert_array_equal(q.obs.total_counts, [4, 4, 8])
    np.testing.assert_array_equal(q.obs.n_genes_by_counts, [2, 2, 1])
    np.testing.assert_allclose(q.obs.pct_counts_mt, [25, 0, 100], rtol=0, atol=1e-12)
    assert q.var.mt.tolist() == [True, False, False]
    assert "mt" not in a.var
    np.testing.assert_array_equal(dense(a.X), before)
    kept = filter_cells(q, min_genes=2, max_genes=2, min_counts=4,
                        max_counts=4, max_pct_mt=25)
    assert kept.obs_names.tolist() == ["cell0", "cell1"]
    assert kept.obs.donor.tolist() == ["a", "a"]
    np.testing.assert_array_equal(dense(kept.X), before[:2])
    assert run_qc(a).obs_names.equals(a.obs_names)


def test_ensembl_current_behavior_and_explicit_symbol_conversion(make_adata):
    a = make_adata([[1, 3]], ["a"], genes=["ENSG1", "ENSG2"])
    a.var["gene_symbols"] = ["MT-CO1", "G1"]
    q = calculate_qc_metrics(a)
    # Known limitation, not an endorsement: gene_symbols is not consulted.
    assert q.var.mt.sum() == 0
    assert q.obs.pct_counts_mt.iloc[0] == 0
    renamed = a.copy()
    renamed.var["ensembl_id"] = renamed.var_names
    renamed.var_names = renamed.var.gene_symbols
    assert calculate_qc_metrics(renamed).obs.pct_counts_mt.iloc[0] == 25
    assert a.var_names.tolist() == ["ENSG1", "ENSG2"]


def test_qc_prefix_and_errors(make_adata):
    a = make_adata([[1, 3]], ["a"], genes=["mt-a", "G1"])
    assert calculate_qc_metrics(a, mitochondrial_prefix="mt-").var.mt.sum() == 1
    with pytest.raises(ValueError, match="missing"):
        filter_cells(a, min_genes=1)
    with pytest.raises(ValueError, match="All cells"):
        run_qc(a, min_counts=999)
    # Current dispatcher passes kwargs to filter_cells only.
    with pytest.raises(TypeError, match="mitochondrial_prefix"):
        run_qc(a, mitochondrial_prefix="mt-")


@pytest.mark.parametrize("csr", [False, True])
@pytest.mark.parametrize("target", [1e4, 5000])
def test_normalization_matches_scanpy_and_preserves_input(make_adata, csr, target):
    a = make_adata([[1, 3, 0], [2, 0, 6]], ["a", "b"], csr=csr)
    before = dense(a.X).copy()
    reference = a.copy()
    sc.pp.normalize_total(reference, target_sum=target)
    sc.pp.log1p(reference)
    out = normalize_expression(a, target_sum=target)
    np.testing.assert_array_equal(dense(out.X), dense(reference.X))
    np.testing.assert_allclose(dense(out.X), np.log1p(before / before.sum(1)[:, None] * target),
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(dense(a.X), before)
    assert "periclock_normalization" not in a.uns
    assert out.uns["periclock_normalization"]["target_sum"] == target


def test_normalization_copy_false(make_adata):
    a = make_adata([[1, 3]], ["a"])
    assert normalize_expression(a, copy=False) is a
    np.testing.assert_allclose(a.X, np.log1p([[2500, 7500]]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("csr", [False, True])
def test_repeat_normalization_currently_proceeds(make_adata, csr):
    a = normalize_expression(make_adata([[1, 9], [7, 3]], ["a", "b"], csr=csr))
    before = dense(a.X).copy()
    reference = a.copy()
    sc.pp.normalize_total(reference, target_sum=1e4)
    sc.pp.log1p(reference)
    again = normalize_expression(a)
    np.testing.assert_array_equal(dense(again.X), dense(reference.X))
    assert not np.allclose(dense(again.X), before)
    np.testing.assert_array_equal(dense(a.X), before)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf])
@pytest.mark.parametrize("csr", [False, True])
def test_invalid_expression_rejected(make_adata, bad, csr):
    a = make_adata([[1, bad]], ["a"], csr=csr)
    with pytest.raises(ValueError, match="negative|NaN or infinite"):
        normalize_expression(a)


def test_empty_normalization():
    with pytest.raises(ValueError, match="empty"):
        normalize_expression(ad.AnnData(np.empty((0, 2))))
