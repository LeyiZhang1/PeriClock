"""Packaging/API additions; Phase 3A scientific tests remain unmodified."""
import importlib
import inspect
import json
import logging
import os
import subprocess
import sys
import tomllib

import pandas as pd
import pytest
from packaging.requirements import Requirement

import periclock
from periclock import build_model_features, get_feature_names, get_model_info, load_model
from periclock import _optional
from test_loader import MODELS


@pytest.mark.parametrize("name,module", [
    ("load_model", "loader"), ("list_models", "loader"),
    ("annotate_cells", "annotation"), ("build_features", "features"),
    ("predict_age", "predictor"),
])
def test_original_public_api_identity_and_signature(name, module):
    original = getattr(importlib.import_module(f"periclock.{module}"), name)
    assert getattr(periclock, name) is original
    assert inspect.signature(getattr(periclock, name)) == inspect.signature(original)
    assert name in periclock.__all__ and name in dir(periclock)


def test_unknown_public_name():
    with pytest.raises(AttributeError, match="does_not_exist"):
        periclock.does_not_exist


@pytest.mark.parametrize("name,parameters,defaults", [
    ("load_model", ["model_name", "model_dir", "model_path"], {"model_name": "PeriClock", "model_dir": None, "model_path": None}),
    ("list_models", [], {}),
    ("build_features", ["adata", "model_name", "individual_key", "celltype_key", "feature_names", "model_bundle", "max_zeros"],
     {"celltype_key": None, "feature_names": None, "model_bundle": None, "max_zeros": 28}),
    ("predict_age", ["X", "model_name", "model_bundle", "actual_age"],
     {"model_name": "PeriClock", "model_bundle": None, "actual_age": None}),
    ("annotate_cells", ["query", "method", "kwargs"], {}),
])
def test_legacy_signature_contract(name, parameters, defaults):
    signature = inspect.signature(getattr(periclock, name))
    assert list(signature.parameters) == parameters
    assert {k: p.default for k, p in signature.parameters.items()
            if p.default is not inspect.Parameter.empty} == defaults
    for key, parameter in signature.parameters.items():
        if key in {"model_name", "individual_key", "celltype_key", "feature_names", "model_bundle", "max_zeros", "actual_age", "method", "model_dir", "model_path"}:
            expected = inspect.Parameter.POSITIONAL_OR_KEYWORD if name == "load_model" else inspect.Parameter.KEYWORD_ONLY
            assert parameter.kind == expected


@pytest.mark.parametrize("name", MODELS)
def test_information_queries(name, bundles):
    b = bundles[name]
    info = get_model_info(name)
    assert info["name"] == name and info["kind"] == b["kind"]
    assert info["n_features"] == len(b["feature_names"])
    assert info["software_version"] == periclock.__version__
    assert info["model_version"] is None
    names = get_feature_names(name)
    assert names == b["feature_names"]
    names.reverse()
    assert get_feature_names(name) == b["feature_names"]


@pytest.mark.parametrize("name", MODELS)
def test_additive_adapter_equals_existing_feature_api(name, bundles, synthetic_query):
    b = bundles[name]
    kwargs = {"model_bundle": b["artifact"]} if b["kind"] == "single_cell" else {
        "feature_names": b["feature_names"], "model_bundle": b}
    expected = periclock.build_features(synthetic_query, model_name=name,
        individual_key="donor", celltype_key="ct", **kwargs)
    actual = build_model_features(synthetic_query, model_bundle=b,
        individual_key="donor", celltype_key="ct")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_adapter_rejects_incomplete_bundle_and_wrong_kind(synthetic_query, bundles):
    with pytest.raises(TypeError, match="wrapper"):
        build_model_features(synthetic_query, model_bundle=None, individual_key="donor")
    with pytest.raises(ValueError, match="wrapper fields"):
        build_model_features(synthetic_query, model_bundle=bundles["PeriClock"]["artifact"], individual_key="donor")
    wrong = dict(bundles["PeriClock"], kind="bulk")
    with pytest.raises(ValueError, match="name and kind"):
        build_model_features(synthetic_query, model_bundle=wrong, individual_key="donor")


def test_model_directory_configuration_precedence(monkeypatch, tmp_path):
    import joblib
    from sklearn.linear_model import LinearRegression
    from periclock import loader
    # Explicit synthetic state, never trained or confused with a release model.
    model = LinearRegression()
    model.coef_ = [1.0]
    model.intercept_ = 0.0
    model.n_features_in_ = 1
    artifact = {"model": model, "selected_features": ["toy"]}
    paths = {}
    for source in ["default", "environment", "explicit", "file"]:
        directory = tmp_path / source
        directory.mkdir()
        paths[source] = directory / "PeriClock.pkl"
        joblib.dump(artifact, paths[source])
    monkeypatch.setattr(loader, "MODEL_DIR", paths["default"].parent)
    monkeypatch.delenv("PERICLOCK_MODEL_DIR", raising=False)
    assert load_model()["path"] == str(paths["default"])
    monkeypatch.setenv("PERICLOCK_MODEL_DIR", str(paths["environment"].parent))
    assert load_model()["path"] == str(paths["environment"])
    # Old positional call is still supported.
    assert load_model("PeriClock", paths["explicit"].parent)["path"] == str(paths["explicit"])
    assert load_model(model_dir=paths["explicit"].parent,
                      model_path=paths["file"])["path"] == str(paths["file"])
    with pytest.raises(FileNotFoundError) as error:
        load_model(model_path=tmp_path / "missing.pkl")
    for hint in ["model_path", "model_dir", "PERICLOCK_MODEL_DIR", "downloaded separately"]:
        assert hint in str(error.value)


@pytest.mark.parametrize("operation,missing,extra", [
    ("celltypist", "celltypist", "celltypist"),
    ("symphony_reference", "symphonypy", "symphony"),
    ("symphony_mapping", "symphonypy", "symphony"),
    ("bbknn", "bbknn", "bbknn"),
])
def test_missing_optional_integration_hint(monkeypatch, operation, missing, extra):
    annotation = importlib.import_module("periclock.annotation")
    real_import = _optional.import_module
    def missing_import(name):
        if name == missing:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name)
    monkeypatch.setattr(_optional, "import_module", missing_import)
    calls = {
        "celltypist": lambda: annotation.annotate_celltypist(None),
        "symphony_reference": lambda: annotation.build_symphony_reference(None, query=None),
        "symphony_mapping": lambda: annotation.annotate_symphony(None, reference=None),
        "bbknn": lambda: annotation.annotate_bbknn_knn(None, reference=None),
    }
    with pytest.raises(ImportError, match=rf"periclock\[{extra}\]"):
        calls[operation]()


def test_broken_transitive_import_is_not_hidden(monkeypatch):
    error = ModuleNotFoundError("broken transitive dependency", name="broken_transitive")
    def broken(name):
        raise error
    monkeypatch.setattr(_optional, "import_module", broken)
    with pytest.raises(ModuleNotFoundError) as captured:
        _optional.require_dependency("celltypist", "celltypist")
    assert captured.value is error


def test_core_imports_and_prediction_without_annotation_packages(release_root, tmp_path, real_model_dir):
    code = r'''
import importlib.abc, sys
blocked = {'scanpy', 'celltypist', 'symphonypy', 'bbknn'}
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('Blocked optional package: '+fullname, name=fullname.split('.')[0])
sys.meta_path.insert(0, BlockOptional())
import periclock
from periclock import load_model, list_models, build_features, build_model_features, predict_age
import numpy as np, pandas as pd
for name in list_models():
    b=load_model(name)
    x=pd.DataFrame(np.zeros((1,len(b['feature_names']))), index=['synthetic'], columns=b['feature_names'])
    result=predict_age(x,model_name=name,model_bundle=b)
    assert np.isfinite(result.Predicted_Age).all()
assert not (blocked & set(sys.modules))
for module in ['qc','normalization','annotation']:
    try:
        __import__('periclock.'+module)
    except ImportError as e:
        assert 'pip install' in str(e) and 'scanpy' in str(e)
    else:
        raise AssertionError('Expected actionable optional dependency error')
print('core-only imports and six predictions passed')
'''
    env = dict(os.environ, PYTHONPATH=str(release_root),
               PERICLOCK_MODEL_DIR=str(real_model_dir))
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, env=env,
                            text=True, capture_output=True, check=True)
    assert "six predictions passed" in result.stdout


def test_logging_is_opt_in(make_adata, caplog):
    from periclock.qc import run_qc
    a = make_adata([[1, 3]], ["a"], genes=["MT-A", "G1"])
    with caplog.at_level(logging.INFO, logger="periclock.qc"):
        run_qc(a)
    assert "Cells before QC: 1" in caplog.text
    assert any(isinstance(h, logging.NullHandler) for h in logging.getLogger("periclock").handlers)


def test_package_metadata_core_and_optional_dependencies(release_root):
    config = tomllib.loads((release_root / "pyproject.toml").read_text())
    project = config["project"]
    assert project["requires-python"] == ">=3.12,<3.13"
    core = {Requirement(r).name for r in project["dependencies"]}
    assert not ({"scanpy", "celltypist", "symphonypy", "bbknn"} & core)
    assert "scikit-learn==1.5.1" in project["dependencies"]
    assert set(project["optional-dependencies"]) == {
        "preprocessing", "celltypist", "symphony", "bbknn", "annotation", "dev", "hub"}
    assert project["optional-dependencies"]["hub"] == ["huggingface-hub==0.25.2"]
    assert (release_root / "requirements.txt").read_text().splitlines()[-1] == "-e ."


def test_annotation_missing_external_data_and_home_path(tmp_path, monkeypatch):
    annotation = importlib.import_module('periclock.annotation')
    with pytest.raises(FileNotFoundError, match='actual .h5ad paths'):
        annotation._load('~/missing-reference.h5ad')
    from pathlib import Path
    resource = tmp_path / 'reference.h5ad'
    resource.touch()
    assert annotation._resource_path(resource) == resource
    monkeypatch.setattr(Path, 'is_file', lambda self: True)
    assert annotation._resource_path('~/reference.h5ad') == Path.home() / 'reference.h5ad'
