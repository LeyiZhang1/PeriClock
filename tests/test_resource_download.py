"""Offline checks for explicit resource downloads; no real model/data downloads."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "resources/download.py"
spec = importlib.util.spec_from_file_location("resource_download", SCRIPT)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


@pytest.fixture
def resource():
    payload = b"offline download verification fixture"
    return payload, {"filename": "fixture.pkl", "size_bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()}


def test_explicit_destination_and_exact_bytes(tmp_path, resource):
    payload, record = resource
    with patch.object(downloader, "urlopen", return_value=io.BytesIO(payload)) as request:
        path = downloader.download(record, "test/model", "commit", tmp_path / "models")
    assert path.read_bytes() == payload
    assert request.call_args.args[0] == "https://huggingface.co/test/model/resolve/commit/fixture.pkl"


def test_corrupt_download_is_removed(tmp_path, resource):
    payload, record = resource
    with patch.object(downloader, "urlopen", return_value=io.BytesIO(b"x" * len(payload))):
        with pytest.raises(ValueError, match="SHA256"):
            downloader.download(record, "test/model", "commit", tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("use_hub", [False, True])
def test_existing_wrong_file_is_not_overwritten(tmp_path, resource, use_hub):
    _, record = resource
    destination = tmp_path / record["filename"]
    destination.write_bytes(b"existing different file")
    with patch.object(downloader, "urlopen") as request:
        with pytest.raises(ValueError, match="Size mismatch"):
            downloader.download(record, "test/model", "commit", tmp_path, use_hub=use_hub)
        request.assert_not_called()
    assert destination.read_bytes() == b"existing different file"


def test_reject_code_repository_as_model_storage(resource):
    _, record = resource
    with pytest.raises(ValueError, match="outside"):
        downloader.download(record, "test/model", "commit", SCRIPT.parent)


def test_dataset_repository_route_and_exact_bytes(tmp_path, resource):
    payload, record = resource
    with patch.object(downloader, "urlopen", return_value=io.BytesIO(payload)) as request:
        path = downloader.download(record, "test/data", "pinned/revision", tmp_path,
                                   repo_type="dataset")
    assert path.read_bytes() == payload
    assert request.call_args.args[0] == (
        "https://huggingface.co/datasets/test/data/resolve/pinned%2Frevision/fixture.pkl"
    )


def test_dataset_download_rejects_checkout(resource):
    _, record = resource
    with patch.object(downloader, "urlopen") as request:
        with pytest.raises(ValueError, match="outside"):
            downloader.download(record, "test/data", "commit", SCRIPT.parent,
                                repo_type="dataset")
        request.assert_not_called()


def test_hub_download_verifies_bytes_and_cleans_temporary_files(tmp_path, resource):
    payload, record = resource
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        path = local_dir / kwargs["filename"]
        path.write_bytes(payload)
        return str(path)

    with patch.dict(sys.modules, huggingface_hub=SimpleNamespace(hf_hub_download=fake_download)):
        path = downloader.download(record, "test/data", "pinned-commit", tmp_path,
                                   repo_type="dataset", use_hub=True)
    assert path.read_bytes() == payload
    assert calls[0]["repo_id"] == "test/data"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == "pinned-commit"
    assert list(tmp_path.iterdir()) == [path]


def test_corrupt_hub_download_is_removed(tmp_path, resource):
    payload, record = resource

    def fake_download(**kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x" * len(payload))
        return str(path)

    with patch.dict(sys.modules, huggingface_hub=SimpleNamespace(hf_hub_download=fake_download)):
        with pytest.raises(ValueError, match="SHA256"):
            downloader.download(record, "test/data", "commit", tmp_path,
                                repo_type="dataset", use_hub=True)
    assert not list(tmp_path.iterdir())


def test_pending_upload_without_revision_fails_before_download(tmp_path):
    catalog = {"resources": [{"id": "pending", "filename": "pending.h5ad",
                "classification": "Pending review", "download_repository_type": "dataset",
                "repo_id": "test/data", "revision": None}]}
    with patch.object(Path, "read_text", return_value=json.dumps(catalog)), \
         patch.object(sys, "argv", ["download.py", "--dataset", "pending",
                                     "--output-dir", str(tmp_path)]), \
         patch.object(downloader, "download") as request:
        with pytest.raises(SystemExit) as error:
            downloader.main()
        assert error.value.code == 2
        request.assert_not_called()
