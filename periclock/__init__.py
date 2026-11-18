"""PeriClock public API. Annotation/QC dependencies are loaded only on demand."""
from importlib import import_module
import logging

from ._version import __version__
from .loader import MODEL_FILES, load_model, list_models, get_model_info, get_feature_names
from .predictor import predict_age

logging.getLogger(__name__).addHandler(logging.NullHandler())

_LAZY_API = {
    "annotate_cells": (".annotation", "annotate_cells"),
    "build_features": (".features", "build_features"),
    "build_model_features": (".api", "build_model_features"),
}


def __getattr__(name):
    if name not in _LAZY_API:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_API[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "MODEL_FILES",
    "load_model",
    "list_models",
    "annotate_cells",
    "build_features",
    "predict_age",
    "__version__",
    "get_model_info",
    "get_feature_names",
    "build_model_features",
]
