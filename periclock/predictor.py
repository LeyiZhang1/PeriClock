"""Age prediction from a feature matrix already built by periclock.features.

This module deliberately does not construct, normalize, scale, impute or
select expression features. It uses the estimator saved by loader.load_model.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

SINGLE_CELL_MODELS = frozenset({
    "PeriClock", "PeriClock-200s", "PeriClock-700s", "PeriClock-1000s"
})
PROPORTION_MODEL = "PeriClock-proportion"
BULK_MODEL = "PeriClock-bulk"
SUPPORTED_MODELS = SINGLE_CELL_MODELS | {PROPORTION_MODEL, BULK_MODEL}


def predict_age(
    X: pd.DataFrame,
    *,
    model_name: str = "PeriClock",
    model_bundle: Mapping[str, Any] | None = None,
    actual_age: pd.Series | Mapping | None = None,
) -> pd.DataFrame:
    """Predict age per Individual from a pre-built feature matrix.

    Parameters
    ----------
    X : pandas.DataFrame
        Output of ``periclock.features.build_features``: Individuals as rows,
        model features as columns, in the trained model's feature order.
    model_name : str
        One of the six published PeriClock model names.
    model_bundle : mapping, optional
        Bundle from ``periclock.loader.load_model(model_name)``. When omitted,
        this function loads it using that existing loader. No joblib calls here.
    actual_age : pandas.Series or mapping, optional
        Chronological age keyed by Individual ID. If supplied, include
        ``Actual_Age`` and ``Age_Diff`` (Predicted_Age - Actual_Age).

    Returns
    -------
    pandas.DataFrame
        Indexed by Individual, containing ``Predicted_Age`` and optionally
        ``Actual_Age`` and ``Age_Diff``.

    Notes
    -----
    No feature engineering or standardization is performed here. In particular,
    the bulk model's saved Pipeline handles scaling internally.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_name: {model_name!r}; choose from {sorted(SUPPORTED_MODELS)}")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be the pandas.DataFrame returned by build_features")
    if X.empty:
        raise ValueError("Feature matrix is empty")
    if not X.index.is_unique or X.index.hasnans:
        raise ValueError("Feature matrix must have unique, nonmissing Individual IDs")
    if not X.columns.is_unique:
        raise ValueError("Feature matrix feature names must be unique")

    if model_bundle is None:
        from .loader import load_model
        model_bundle = load_model(model_name)
    if not isinstance(model_bundle, Mapping):
        raise TypeError("model_bundle must be the dictionary returned by load_model")
    if "model" not in model_bundle:
        raise KeyError("Model bundle has no saved 'model' estimator")

    estimator = model_bundle["model"]
    if not hasattr(estimator, "predict"):
        raise TypeError("Saved 'model' does not provide predict()")

    # loader.load_model() returns a wrapper whose single-cell training artifact
    # contains selected_features. Also accept the original artifact itself,
    # preserving the previously tested direct-artifact prediction interface.
    if model_name in SINGLE_CELL_MODELS:
        artifact = model_bundle.get("artifact")
        if isinstance(artifact, Mapping):
            expected = artifact.get("selected_features")
        else:
            expected = model_bundle.get("selected_features")
        if expected is None:
            raise KeyError("Single-cell model bundle has no 'artifact[selected_features]' or 'selected_features'")
    else:
        expected = model_bundle.get("feature_names")
        if expected is None:
            raise KeyError("Model bundle has no 'feature_names'")
    expected = list(expected)
    if X.columns.tolist() != expected:
        missing = [name for name in expected if name not in X.columns]
        extra = [name for name in X.columns if name not in expected]
        if missing or extra:
            raise ValueError(
                f"Feature matrix does not match {model_name}: missing={missing[:15]}, extra={extra[:15]}"
            )
        raise ValueError("Feature columns are not in the original training order; use build_features output")

    values = X.to_numpy()
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("Feature matrix must contain only finite numerical values")

    # The original single-cell code called model.predict(X_selected), where
    # X_selected is a NumPy array. For bulk the full saved Pipeline was fitted
    # on a DataFrame of gene names; do not bypass its StandardScaler.
    model_input = values if model_name in SINGLE_CELL_MODELS else X
    predicted = np.asarray(estimator.predict(model_input)).reshape(-1)
    if predicted.size != len(X) or not np.isfinite(predicted).all():
        raise ValueError("Model returned invalid predicted ages")

    result = pd.DataFrame({"Predicted_Age": predicted}, index=X.index.copy())
    if actual_age is not None:
        if isinstance(actual_age, pd.Series):
            if not actual_age.index.is_unique:
                raise ValueError("actual_age index must be unique")
            ages = actual_age.reindex(X.index)
        elif isinstance(actual_age, Mapping):
            ages = pd.Series(dict(actual_age)).reindex(X.index)
        else:
            raise TypeError("actual_age must be a pandas.Series or an Individual-to-age mapping")
        ages = pd.to_numeric(ages, errors="raise")
        if ages.isna().any() or not np.isfinite(ages.to_numpy(dtype=float)).all():
            raise ValueError("actual_age is missing or nonfinite for at least one Individual")
        result.insert(0, "Actual_Age", ages.to_numpy(dtype=float))
        result["Age_Diff"] = result["Predicted_Age"] - result["Actual_Age"]
    return result
