"""Additive wrapper adapter; existing feature and prediction APIs stay unchanged."""
from collections.abc import Mapping

from .features import build_features
from .loader import MODEL_FILES


def build_model_features(adata, *, model_bundle, individual_key,
                         celltype_key=None, max_zeros=28):
    """Build inputs from a loader wrapper using its model-specific saved state.

    model_bundle must be the wrapper returned by load_model(). Existing
    build_features() continues to accept its original model-specific arguments.
    This adapter performs no normalization, annotation, fitting or extra filling.
    """
    if not isinstance(model_bundle, Mapping):
        raise TypeError("model_bundle must be the wrapper returned by load_model()")
    required = {"name", "kind", "model", "artifact", "feature_names"}
    if not required.issubset(model_bundle):
        raise ValueError("model_bundle must include the loader wrapper fields: " + ", ".join(sorted(required)))
    name = model_bundle["name"]
    if name not in MODEL_FILES:
        raise ValueError(f"Unknown model in loader wrapper: {name!r}")
    expected_kind = "bulk" if name == "PeriClock-bulk" else (
        "proportion" if name == "PeriClock-proportion" else "single_cell")
    if model_bundle["kind"] != expected_kind:
        raise ValueError("Loader wrapper name and kind do not match")
    kwargs = {"model_bundle": model_bundle["artifact"]} if expected_kind == "single_cell" else {
        "feature_names": model_bundle["feature_names"], "model_bundle": model_bundle}
    return build_features(adata, model_name=name, individual_key=individual_key,
                          celltype_key=celltype_key, max_zeros=max_zeros, **kwargs)
