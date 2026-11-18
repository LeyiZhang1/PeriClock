import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.preprocessing import StandardScaler

from periclock.features import build_features
from reference import full_column_reference, historical_coverage

SINGLE = ["PeriClock", "PeriClock-200s", "PeriClock-700s", "PeriClock-1000s"]
TYPES = ["T", "T_sub"] + [f"ct{i}" for i in range(2, 48)]


def toy_bundle():
    """Explicit synthetic transformer state; nothing is fitted or trained."""
    expression = ["T_G1", "T_G_2", "T_sub_G1", "T_sub_MISSING", "ct2_G1"]
    vt = VarianceThreshold(threshold=1e-5)
    vt.variances_ = np.array([1, 1, 1, 1, 0.0])
    vt.n_features_in_ = 5
    vt.feature_names_in_ = np.array(expression, object)
    skb = SelectKBest(k=3)
    skb.scores_ = np.array([1, 0, 2, 3.0])
    skb.n_features_in_ = 4

    def scaler(mean, scale):
        result = StandardScaler()
        result.mean_ = np.asarray(mean)
        result.scale_ = np.asarray(scale)
        result.var_ = result.scale_ ** 2
        result.n_features_in_ = len(mean)
        result.n_samples_seen_ = 1
        return result

    return {"vt": vt, "skb": skb, "scaler_prop": scaler([0.1]*48, [2.0]*48),
            "scaler_expr": scaler([1., 2., 3.], [2., 4., 5.]),
            "prop_cols": [f"{t}_ratio" for t in TYPES], "expr_cols": expression,
            "selected_features": ["T_sub_MISSING", "T_ratio", "T_sub_G1", "T_G1"]}


@pytest.mark.parametrize("csr", [False, True])
def test_analytic_single_cell_features_missing_types_genes_and_scaling(make_adata, csr):
    a = make_adata([[2, 4], [6, 8], [10, 12], [14, 16], [20, 22]],
                   ["a"]*4+["b"], ["T", "T", "T_sub", "unknown", "T"],
                   ["G1", "G_2"], csr=csr)
    b = toy_bundle()
    with pytest.warns(UserWarning, match="1 expression genes absent"):
        x = build_features(a, model_name="PeriClock", individual_key="donor", celltype_key="ct",
                           model_bundle=b, max_zeros=None)
    assert x.columns.tolist() == b["selected_features"]
    assert x.index.tolist() == ["a", "b"]
    # Missing raw values are zero BEFORE scaling, not after scaling.
    np.testing.assert_allclose(x, [[-.6, .2, 2, 1.5], [-.6, .45, -.5, 9.5]], rtol=0, atol=1e-12)


def test_proportions_denominator_and_absent_types(make_adata):
    a = make_adata([[1]]*5, ["a"]*4+["b"], ["T", "T", "T_sub", "unknown", "T"])
    x = build_features(a, model_name="PeriClock-proportion", individual_key="donor",
                       celltype_key="ct", feature_names=TYPES, max_zeros=None)
    np.testing.assert_array_equal(x.loc["a", ["T", "T_sub", "ct2"]], [.5, .25, 0])
    assert x.loc["a"].sum() == .75
    assert x.loc["b", "T"] == 1


@pytest.mark.parametrize("n,eligible", [(19, False), (20, True), (21, True)])
def test_current_coverage_boundary(make_adata, n, eligible):
    a = make_adata([[1]]*n, ["a"]*n, TYPES[:n])
    kwargs = dict(model_name="PeriClock-proportion", individual_key="donor",
                  celltype_key="ct", feature_names=TYPES)
    if eligible:
        assert build_features(a, **kwargs).index.tolist() == ["a"]
    else:
        with pytest.raises(ValueError, match="No individuals meet"):
            build_features(a, **kwargs)


def test_partial_donor_exclusion(make_adata):
    labels = TYPES[:19] + TYPES[:20]
    a = make_adata([[1]]*39, ["low"]*19+["keep"]*20, labels)
    x = build_features(a, model_name="PeriClock-proportion", individual_key="donor",
                       celltype_key="ct", feature_names=TYPES)
    assert x.index.tolist() == ["keep"]


def test_observed_label_coverage_is_not_fixed_model_coverage(make_adata):
    # Historical observed-table coverage keeps this donor; fixed-48 excludes it.
    a = make_adata([[1]]*19, ["a"]*19, TYPES[:19])
    obs = a.obs.assign(age=30)
    assert historical_coverage(obs).index.tolist() == ["a_0"]
    with pytest.raises(ValueError, match="No individuals meet"):
        build_features(a, model_name="PeriClock-proportion", individual_key="donor",
                       celltype_key="ct", feature_names=TYPES)


def test_age_grouping_difference_is_explicit(make_adata):
    a = make_adata([[1], [2]], ["a", "a"], ["T", "T"])
    assert historical_coverage(a.obs.assign(age=[30, 40])).index.tolist() == ["a_0", "a_1"]
    x = build_features(a, model_name="PeriClock-proportion", individual_key="donor",
                       celltype_key="ct", feature_names=TYPES, max_zeros=None)
    assert x.index.tolist() == ["a"]


@pytest.mark.parametrize("csr", [False, True])
def test_bulk_analytic_mean_order_and_no_celltype_requirement(make_adata, csr):
    a = make_adata([[1, 2], [3, 6], [10, 20]], ["b", "b", "a"],
                   genes=["G1", "G2"], csr=csr)
    x = build_features(a, model_name="PeriClock-bulk", individual_key="donor", feature_names=["G2", "G1"])
    assert x.index.tolist() == ["a", "b"]
    np.testing.assert_array_equal(x, [[20, 10], [4, 2]])


def test_bulk_missing_gene_training_mean_and_errors(make_adata):
    a = make_adata([[2], [6]], ["a", "a"], genes=["G1"])
    scaler = StandardScaler()
    scaler.mean_ = np.array([100., 200.])
    class Pipeline:
        named_steps = {"standardscaler": scaler}
    kwargs = dict(model_name="PeriClock-bulk", individual_key="donor", feature_names=["G1", "missing"])
    with pytest.raises(ValueError, match="Missing 1 required bulk genes"):
        build_features(a, **kwargs)
    with pytest.warns(UserWarning, match="Imputed 1 missing bulk genes"):
        x = build_features(a, **kwargs, model_bundle={"model": Pipeline()})
    np.testing.assert_array_equal(x, [[4, 200]])
    with pytest.raises(ValueError, match="No required bulk genes"):
        build_features(a, model_name="PeriClock-bulk", individual_key="donor", feature_names=["absent"])


def test_bulk_training_subset_renormalization_can_differ(make_adata):
    # bulk_model.ipynb cell 4 renormalizes the ratio columns supplied to it.
    # If only T is represented there, an unknown type's expression is omitted.
    a = make_adata([[2], [10]], ["a", "a"], ["T", "unknown"], ["G1"])
    current = build_features(a, model_name="PeriClock-bulk", individual_key="donor", feature_names=["G1"])
    historical_subset_value = 2 * (.5 / .5)
    assert current.loc["a", "G1"] == 6
    assert historical_subset_value == 2
    assert current.loc["a", "G1"] != historical_subset_value


@pytest.mark.parametrize("name", SINGLE)
def test_real_saved_transforms_against_full_column_reference(name, bundles, synthetic_query):
    a = bundles[name]["artifact"]
    expected = full_column_reference(synthetic_query, a)
    with pytest.warns(UserWarning, match="expression genes absent"):
        actual = build_features(synthetic_query, model_name=name, individual_key="donor",
                                celltype_key="ct", model_bundle=a)
    pd.testing.assert_index_equal(actual.index, expected.index)
    pd.testing.assert_index_equal(actual.columns, expected.columns)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_wrapper_mismatch_is_characterized(bundles, synthetic_query):
    with pytest.raises(ValueError, match="lacks original preprocessing fields"):
        build_features(synthetic_query, model_name="PeriClock", individual_key="donor",
                       celltype_key="ct", model_bundle=bundles["PeriClock"])
