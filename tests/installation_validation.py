"""Explicit offline installation checks; no network or global environment writes.

Run: python -m pytest tests/installation_validation.py
Temporary virtual environments reuse verified system-site dependencies. This is
not a clean download/resolver test. Subprocess imports never insert the checkout
into sys.path and use isolated Python mode from unrelated working directories.
"""
import email
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import venv
import zipfile

import pytest


def offline_env():
    env = dict(os.environ, PIP_NO_INDEX="1", PIP_NO_BUILD_ISOLATION="0",
               PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_CACHE_DIR="/tmp/periclock-3b-pip-cache")
    env.pop("PYTHONPATH", None)
    env.pop("PERICLOCK_MODEL_DIR", None)
    return env


@pytest.fixture(scope="module")
def archives(tmp_path_factory, release_root):
    output = tmp_path_factory.mktemp("periclock-distributions")
    env = offline_env()
    subprocess.run([sys.executable, "-B", "-m", "pip", "wheel", "--no-deps",
                    "--no-build-isolation", "--no-index", "--wheel-dir", str(output), str(release_root)],
                   cwd=output, env=env, check=True, capture_output=True, text=True)
    code = "from setuptools.build_meta import build_sdist; import sys; build_sdist(sys.argv[1])"
    subprocess.run([sys.executable, "-B", "-c", code, str(output)], cwd=release_root,
                   env=env, check=True, capture_output=True, text=True)
    return {"wheel": next(output.glob("*.whl")), "sdist": next(output.glob("*.tar.gz"))}


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_distribution_excludes_research_assets(archives, kind):
    if kind == "wheel":
        with zipfile.ZipFile(archives[kind]) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(archives[kind]) as archive:
            names = archive.getnames()
    assert any(n.endswith("periclock/_version.py") for n in names)
    assert any(n.endswith("periclock/features.py") for n in names)
    assert not any(n.endswith((".pkl", ".joblib", ".h5ad", ".ipynb", ".pyc")) for n in names)
    assert not any(part in {"models", "examples", "tests", "results", "__pycache__"}
                   for n in names for part in Path(n).parts)
    assert not any(Path(n).name in {"RELEASE_AUDIT.md", "SCIENTIFIC_BASELINE.md", "TEST_REPORT.md"}
                   for n in names)


def test_source_archive_builds_a_wheel(archives, tmp_path):
    subprocess.run([sys.executable, "-B", "-m", "pip", "wheel", "--no-deps",
                    "--no-build-isolation", "--no-index", "--wheel-dir", str(tmp_path), str(archives["sdist"])],
                   cwd=tmp_path, env=offline_env(), check=True, capture_output=True, text=True)
    with zipfile.ZipFile(next(tmp_path.glob("*.whl"))) as rebuilt, zipfile.ZipFile(archives["wheel"]) as original:
        for name in original.namelist():
            if name.startswith("periclock/") and name.endswith(".py"):
                assert rebuilt.read(name) == original.read(name)


@pytest.fixture(scope="module", params=["editable", "wheel"])
def installed(request, archives, tmp_path_factory, release_root):
    root = tmp_path_factory.mktemp(f"periclock-{request.param}")
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(root / "venv")
    python = root / "venv/bin/python"
    env = offline_env()
    command = [str(python), "-B", "-m", "pip", "install"]
    if request.param == "editable":
        command += ["-e", "."]
        cwd = release_root
    else:
        command += [str(archives["wheel"])]
        cwd = root
    # Offline environment variables disable build isolation; dependencies are
    # still checked by pip (no --no-deps for installation).
    result = subprocess.run(command, cwd=cwd, env=env, check=True, capture_output=True, text=True)
    (root / "installation.log").write_text(result.stdout + result.stderr)
    return {"mode": request.param, "python": python, "root": root, "env": env}


def run_installed(installed, code, cwd=None, args=(), model_dir=None):
    env = dict(installed["env"])
    if model_dir is not None:
        env["PERICLOCK_MODEL_DIR"] = str(model_dir)
    return subprocess.run([str(installed["python"]), "-I", "-B", "-c", code, *map(str, args)],
                          cwd=cwd or installed["root"], env=env,
                          capture_output=True, text=True, check=True)


@pytest.mark.parametrize("directory", ["first", "second/nested", "third"])
def test_installed_public_imports_from_unrelated_directories(installed, release_root, directory):
    cwd = installed["root"] / directory
    cwd.mkdir(parents=True, exist_ok=True)
    result = run_installed(installed, """
import json, periclock, importlib.metadata as m
from periclock import load_model, list_models
assert len(list_models())==6
assert periclock.__version__==m.version('periclock')=='0.1.0.dev0'
print(json.dumps({'path':periclock.__file__}))
""", cwd=cwd)
    actual = Path(json.loads(result.stdout)["path"]).resolve()
    if installed["mode"] == "editable":
        assert actual == release_root / "periclock/__init__.py"
    else:
        assert actual.is_relative_to(installed["root"] / "venv")
        assert not actual.is_relative_to(release_root)


def test_installed_six_models_and_prediction(installed, real_model_dir):
    code = """
import periclock, numpy as np, pandas as pd
from periclock import load_model, list_models, get_model_info, get_feature_names, predict_age
for name in list_models():
    b=load_model(name)
    assert get_feature_names(name)==b['feature_names']
    assert get_model_info(name)['n_features']==len(b['feature_names'])
    x=pd.DataFrame(np.zeros((1,len(b['feature_names']))),index=['synthetic'],columns=b['feature_names'])
    y=predict_age(x,model_name=name,model_bundle=b)
    assert y.index.tolist()==['synthetic'] and np.isfinite(y.Predicted_Age).all()
print('six models passed')
"""
    result = run_installed(installed, code, model_dir=real_model_dir)
    assert "six models passed" in result.stdout


def test_installed_core_prediction_without_optional_imports(installed, real_model_dir):
    code = """
import importlib.abc, sys
blocked={'scanpy','celltypist','symphonypy','bbknn'}
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError(fullname,name=fullname.split('.')[0])
sys.meta_path.insert(0,BlockOptional())
from periclock import load_model, list_models, build_model_features, predict_age
import numpy as np, pandas as pd
for name in list_models():
    b=load_model(name)
    x=pd.DataFrame(np.zeros((1,len(b['feature_names']))),columns=b['feature_names'])
    assert np.isfinite(predict_age(x,model_name=name,model_bundle=b).Predicted_Age).all()
assert not (blocked & set(sys.modules))
"""
    run_installed(installed, code, model_dir=real_model_dir)


def test_installed_dependency_metadata(installed):
    run_installed(installed, """
from importlib.metadata import distribution, version
from packaging.requirements import Requirement
dist=distribution('periclock')
reqs=[Requirement(r) for r in dist.requires]
core={r.name for r in reqs if r.marker is None}
for requirement in reqs:
    if requirement.marker is None:
        assert requirement.specifier.contains(version(requirement.name)), str(requirement)
assert not ({'scanpy','celltypist','symphonypy','bbknn'} & core)
assert 'scikit-learn' in core
assert set(dist.metadata.get_all('Provides-Extra'))=={'preprocessing','celltypist','symphony','bbknn','annotation','dev'}
""")


def test_wheel_missing_resources_have_actionable_error(archives, tmp_path):
    # Inspect package code via wheel in a fresh subprocess; no source-root import.
    code = """
import sys
sys.path.insert(0,sys.argv[1])
from periclock import load_model
try: load_model()
except FileNotFoundError as e:
    assert 'model_dir' in str(e) and 'PERICLOCK_MODEL_DIR' in str(e)
else: raise AssertionError('Wheel must not bundle model files')
"""
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(archives["wheel"])],
                   cwd=tmp_path, env=offline_env(), capture_output=True, text=True, check=True)
