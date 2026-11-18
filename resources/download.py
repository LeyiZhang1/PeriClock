"""Explicit, checksum-verified resource downloads; never runs during package import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote
from urllib.request import urlopen


def verify(path, record):
    path = Path(path)
    if path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"Size mismatch: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != record["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")


def download(record, repo_id, revision, output_dir, repo_type="model", use_hub=False):
    if repo_type not in {"model", "dataset"}:
        raise ValueError("Repository type must be model or dataset")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise ValueError("Use an actual Hugging Face repository ID: owner/name")
    if not revision.strip():
        raise ValueError("Specify the published revision, preferably its commit hash")
    filename = record["filename"]
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("Invalid resource filename in catalog")
    output_dir = Path(output_dir).expanduser().resolve()
    release_root = Path(__file__).resolve().parents[1]
    if output_dir == release_root or release_root in output_dir.parents:
        raise ValueError("Choose resource storage outside the GitHub code repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    if destination.exists():
        verify(destination, record)
        return destination
    if use_hub:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "Authenticated downloads require: python -m pip install '.[hub]'. "
                "Authenticate with huggingface-cli login or HF_TOKEN for private repositories."
            ) from error
        # SDK authentication uses the existing login; no credentials are changed.
        with tempfile.TemporaryDirectory(dir=output_dir, prefix=".periclock-hub-") as staging:
            staging = Path(staging)
            downloaded = Path(hf_hub_download(
                repo_id=repo_id, filename=filename, repo_type=repo_type,
                revision=revision, local_dir=staging / "files",
                cache_dir=staging / "cache",
            )).resolve()
            if staging.resolve() not in downloaded.parents:
                raise ValueError("Hub download returned a path outside temporary resource storage")
            verify(downloaded, record)
            if destination.exists():
                verify(destination, record)
                return destination
            os.replace(downloaded, destination)
        return destination
    prefix = "datasets/" if repo_type == "dataset" else ""
    url = f"https://huggingface.co/{prefix}{repo_id}/resolve/{quote(revision, safe='')}/{quote(filename, safe='')}"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=f".{filename}.", suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            total = 0
            with urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > record["size_bytes"]:
                        raise ValueError(f"Download exceeds expected size: {filename}")
                    stream.write(chunk)
        verify(temporary, record)
        if destination.exists():
            verify(destination, record)
            return destination
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List models, planned datasets and status without network access")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", help="A supported model name, or all")
    selection.add_argument("--dataset", help="One dataset resource ID from --list; download only after publication")
    parser.add_argument("--repo-id", help="Override the verified repository ID in the catalog")
    parser.add_argument("--revision", help="Override the verified commit in the catalog")
    parser.add_argument("--use-hub", action="store_true", help="Use the optional Hub SDK and existing login; automatic for private resources")
    parser.add_argument("--output-dir", type=Path, help="User-chosen resource storage outside the code repository")
    args = parser.parse_args()
    catalog = json.loads(Path(__file__).with_name("catalog.json").read_text())
    models = [r for r in catalog["resources"] if r["classification"] == "HF Models"]
    datasets = [r for r in catalog["resources"] if r.get("download_repository_type") == "dataset"]
    if args.list:
        for record in models + datasets:
            kind = "dataset" if record in datasets else "model"
            print(f'{kind} {record["id"]} ({record["filename"]}): {record["size_bytes"]} bytes; {record["publication_status"]}')
        return
    if not all([args.model or args.dataset, args.output_dir]):
        parser.error("Download requires --model or --dataset and an external --output-dir")
    pool = datasets if args.dataset else models
    name = args.dataset or args.model
    chosen = models if args.model == "all" else [r for r in pool if r["id"] == name]
    if not chosen:
        parser.error("Unknown resource; use --list")
    jobs = []
    for record in chosen:
        repo_id = args.repo_id or record.get("repo_id") or record.get("repository_id")
        revision = args.revision or record.get("revision")
        if not repo_id or not revision:
            parser.error(f'{record["id"]}: no verified uploaded revision; upload is pending. '
                         "Check RESOURCE_DOWNLOAD.md or specify a confirmed --revision.")
        jobs.append((record, repo_id, revision))
    for record, repo_id, revision in jobs:
        print(download(record, repo_id, revision, args.output_dir,
                       repo_type="dataset" if args.dataset else "model",
                       use_hub=args.use_hub or record.get("requires_authentication", False)))


if __name__ == "__main__":
    main()
