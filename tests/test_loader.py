import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from periclock import list_models, load_model

MODELS = ["PeriClock", "PeriClock-200s", "PeriClock-700s", "PeriClock-1000s",
          "PeriClock-bulk", "PeriClock-proportion"]


@pytest.mark.parametrize("name", MODELS)
def test_saved_model_signature_and_immutable_file(name, bundles, release_root):
    manifest = json.loads((release_root / "tests/baselines/artifacts.json").read_text())
    b = bundles[name]
    names = b["feature_names"]
    assert len(names) == len(set(names)) == b["model"].n_features_in_
    assert list(b["model"].feature_names_in_) == names
    record = manifest[name]
    with Path(b["path"]).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]
    encoded = json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == record["feature_order_sha256"]
    a = b["artifact"]
    field = a["model_info"]["feature_names"] if b["kind"] == "bulk" else (
        a["feature_names"] if b["kind"] == "proportion" else a["selected_features"])
    assert names == list(field)
    if b["kind"] == "single_cell":
        vt = np.asarray(a["expr_cols"], object)[a["vt"].get_support()]
        selected = vt[a["skb"].get_support()].tolist()
        assert len(selected) == a["scaler_expr"].n_features_in_ == 8000
        assert list(a["scaler_prop"].feature_names_in_) == a["prop_cols"]
        if "selected_expr_cols" in a:
            assert selected == a["selected_expr_cols"]


def test_model_registry():
    assert list_models() == MODELS


def test_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        load_model("not-a-model")


def test_missing_model_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model not found"):
        load_model(model_dir=tmp_path)


def test_model_diagnostic_script(release_root, real_model_dir):
    result = subprocess.run([sys.executable, "-B", str(release_root / "check_models.py")],
                            cwd="/tmp", env=dict(os.environ, PERICLOCK_MODEL_DIR=str(real_model_dir)),
                            text=True, capture_output=True, check=True)
    for name in MODELS:
        assert f"{name}:" in result.stdout
    assert result.stdout.count("Recovered pre-scaled expression features: 8000") == 4


def test_model_diagnostic_missing_resources(release_root, tmp_path):
    result = subprocess.run(
        [sys.executable, '-B', str(release_root / 'check_models.py'),
         '--model-dir', str(tmp_path)], text=True, capture_output=True)
    assert result.returncode != 0
    assert not result.stdout
    for hint in ['model_path', 'model_dir', 'PERICLOCK_MODEL_DIR']:
        assert hint in result.stderr
