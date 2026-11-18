from pathlib import Path
import logging
import os
import joblib

from ._version import __version__

logger = logging.getLogger(__name__)


# ============================================================
# 1. Model registry
# ============================================================
MODEL_FILES = {
    "PeriClock": "PeriClock.pkl",
    "PeriClock-200s": "PeriClock-200s.pkl",
    "PeriClock-700s": "PeriClock-700s.pkl",
    "PeriClock-1000s": "PeriClock-1000s.pkl",
    "PeriClock-bulk": "PeriClock-bulk.pkl",
    "PeriClock-proportion": "PeriClock-proportion.pkl",
}

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


# ============================================================
# 2. Load model
# ============================================================
def load_model(model_name="PeriClock", model_dir=None, model_path=None):

    """Load a trusted external artifact.

    Resolution: model_path > model_dir > PERICLOCK_MODEL_DIR > models/.
    The default location is retained for compatibility; assets are not bundled.
    """
    if model_name not in MODEL_FILES:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_FILES)}"
        )

    # Priority: model_path > model_dir > environment > default
    if model_path is not None:
        model_path = Path(model_path).expanduser()

    else:
        configured_dir = model_dir if model_dir is not None else os.environ.get("PERICLOCK_MODEL_DIR")
        model_dir = Path(configured_dir).expanduser() if configured_dir else MODEL_DIR
        model_path = model_dir / MODEL_FILES[model_name]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Models are downloaded separately. "
            "Specify model_path='/path/to/model.pkl', model_dir='/path/to/models', "
            "or set PERICLOCK_MODEL_DIR to the downloaded model directory."
        )

    logger.debug("Loading %s from %s", model_name, model_path)
    artifact = joblib.load(model_path)

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Invalid model artifact structure")

    model = artifact["model"]

    if not hasattr(model, "predict"):
        raise ValueError("Loaded model does not support prediction")

    if model_name == "PeriClock-bulk":
        kind = "bulk"
        feature_names = artifact["model_info"]["feature_names"]

    elif model_name == "PeriClock-proportion":
        kind = "proportion"
        feature_names = artifact["feature_names"]

    else:
        kind = "single_cell"
        feature_names = artifact["selected_features"]

    feature_names = list(feature_names)

    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: "
                f"model expects {model.n_features_in_}, "
                f"but artifact contains {len(feature_names)}"
            )

    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != feature_names:
            raise ValueError(
                "Feature order does not match the trained model"
            )

    return {
        "name": model_name,
        "kind": kind,
        "model": model,
        "artifact": artifact,
        "feature_names": feature_names,
        "path": str(model_path),
    }


# ============================================================
# 3. List available models
# ============================================================
def list_models():
    return list(MODEL_FILES.keys())


def get_feature_names(model_name="PeriClock", model_dir=None):
    """Return a new list of final estimator input names in saved training order."""
    return list(load_model(model_name, model_dir=model_dir)["feature_names"])


def get_model_info(model_name="PeriClock", model_dir=None):
    """Inspect a trusted saved model without fitting or changing its state.

    Model release versions are not recorded uniformly; software_version must not
    be interpreted as a version or a performance claim for the saved estimator.
    """
    bundle = load_model(model_name, model_dir=model_dir)
    model = bundle["model"]
    return {
        "name": bundle["name"],
        "kind": bundle["kind"],
        "n_features": len(bundle["feature_names"]),
        "estimator": f"{type(model).__module__}.{type(model).__name__}",
        "path": bundle["path"],
        "software_version": __version__,
        "model_version": None,
    }
