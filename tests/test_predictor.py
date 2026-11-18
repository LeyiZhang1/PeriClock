import numpy as np
import pandas as pd
import pytest

from periclock import build_features, predict_age
from test_loader import MODELS


@pytest.fixture(params=MODELS)
def model_input(request, bundles, synthetic_query):
    name = request.param
    b = bundles[name]
    kwargs = {"model_bundle": b["artifact"]} if b["kind"] == "single_cell" else {
        "feature_names": b["feature_names"], "model_bundle": b}
    if b["kind"] == "single_cell":
        with pytest.warns(UserWarning, match="expression genes absent"):
            x = build_features(synthetic_query, model_name=name, individual_key="donor", celltype_key="ct", **kwargs)
    elif b["kind"] == "bulk":
        with pytest.warns(UserWarning, match="Imputed .* missing bulk genes"):
            x = build_features(synthetic_query, model_name=name, individual_key="donor", **kwargs)
    else:
        x = build_features(synthetic_query, model_name=name, individual_key="donor", celltype_key="ct", **kwargs)
    return name, b, x


def test_saved_estimator_prediction_without_extra_transforms(model_input):
    name, b, x = model_input
    result = predict_age(x, model_name=name, model_bundle=b)
    assert result.columns.tolist() == ["Predicted_Age"]
    pd.testing.assert_index_equal(result.index, x.index)
    model = b["model"]
    if b["kind"] == "single_cell":
        # Independent linear formula; no wrapper call or estimator predict.
        expected = x.to_numpy() @ model.coef_ + model.intercept_
    elif b["kind"] == "bulk":
        scaler = model.named_steps["standardscaler"]
        linear = model.named_steps["elasticnetcv"]
        expected = ((x.to_numpy()-scaler.mean_) / scaler.scale_) @ linear.coef_ + linear.intercept_
    else:
        expected = model.predict(x)
    np.testing.assert_allclose(result.Predicted_Age, expected, rtol=0, atol=1e-12)


def test_age_alignment_and_difference(model_input):
    name, b, x = model_input
    ages = pd.Series([60., 30.], index=["donor_b", "donor_a"])
    result = predict_age(x, model_name=name, model_bundle=b, actual_age=ages)
    np.testing.assert_array_equal(result.Actual_Age, [30., 60.])
    np.testing.assert_array_equal(result.Age_Diff, result.Predicted_Age - result.Actual_Age)


def test_single_donor_and_automatic_loading(model_input):
    name, b, x = model_input
    one = predict_age(x.iloc[:1], model_name=name)
    many = predict_age(x, model_name=name, model_bundle=b)
    np.testing.assert_allclose(one.Predicted_Age, many.Predicted_Age.iloc[:1], rtol=0, atol=1e-12)


def test_reordered_features_are_rejected(model_input):
    name, b, x = model_input
    with pytest.raises(ValueError, match="original training order"):
        predict_age(x.iloc[:, ::-1], model_name=name, model_bundle=b)


@pytest.mark.parametrize("case", ["missing", "extra", "nan", "duplicate_donor", "missing_age", "nonfinite_age"])
def test_invalid_prediction_inputs(model_input, case):
    name, b, x = model_input
    x = x.copy()
    kwargs = {}
    if case == "missing":
        x = x.iloc[:, 1:]
    elif case == "extra":
        x["not-a-feature"] = 1
    elif case == "nan":
        x.iloc[0, 0] = np.nan
    elif case == "duplicate_donor":
        x.index = ["same"]*len(x)
    elif case == "missing_age":
        kwargs["actual_age"] = {"donor_a": 30}
    else:
        kwargs["actual_age"] = {"donor_a": 30, "donor_b": np.inf}
    with pytest.raises(ValueError):
        predict_age(x, model_name=name, model_bundle=b, **kwargs)


@pytest.fixture
def analytic_input():
    """Independent synthetic estimator; no external model and no fitting."""
    class Estimator:
        def predict(self, x):
            return np.asarray(x) @ np.array([2., -3.]) + 5.
    x = pd.DataFrame([[1., 2.], [4., 1.]], index=['a', 'b'], columns=['f1', 'f2'])
    return x, {'model': Estimator(), 'selected_features': ['f1', 'f2']}


def test_analytic_prediction_and_age_alignment(analytic_input):
    x, bundle = analytic_input
    result = predict_age(x, model_bundle=bundle, actual_age={'b': 20., 'a': 10.})
    np.testing.assert_array_equal(result.Predicted_Age, [1., 10.])
    np.testing.assert_array_equal(result.Actual_Age, [10., 20.])
    np.testing.assert_array_equal(result.Age_Diff, [-9., -10.])


@pytest.mark.parametrize('case', ['missing', 'extra', 'nan', 'duplicate_donor',
                                  'missing_age', 'nonfinite_age', 'reordered'])
def test_analytic_prediction_input_validation(analytic_input, case):
    x, bundle = analytic_input
    kwargs = {}
    if case == 'missing':
        x = x.iloc[:, 1:]
    elif case == 'extra':
        x['extra'] = 1.
    elif case == 'nan':
        x.iloc[0, 0] = np.nan
    elif case == 'duplicate_donor':
        x.index = ['same', 'same']
    elif case == 'missing_age':
        kwargs['actual_age'] = {'a': 10.}
    elif case == 'nonfinite_age':
        kwargs['actual_age'] = {'a': 10., 'b': np.inf}
    else:
        x = x.iloc[:, ::-1]
    with pytest.raises(ValueError):
        predict_age(x, model_bundle=bundle, **kwargs)
