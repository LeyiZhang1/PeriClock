# Installation

Use Python **3.12**. The dependencies in `pyproject.toml` retain the supported
saved-model versions, including scikit-learn 1.5.1.

```bash
python -m pip install -e .
# Test suite and preprocessing:
python -m pip install -e '.[dev,preprocessing]'
# Select annotation capabilities as needed:
python -m pip install -e '.[celltypist]'  # or .[symphony] / .[bbknn]
python -c 'import periclock; print(periclock.__version__)'
```

`python -m pip install -r requirements.txt` installs the explicitly pinned core
dependencies and this checkout in editable mode. Its pins match the project
dependencies in `pyproject.toml`; optional annotation and test dependencies remain
available through the extras above. No setup.py is required.

For an offline install with compatible build/runtime dependencies already present:

```bash
python -m pip install --no-index --no-build-isolation -e .
```

This is not a fresh dependency-resolution test; build isolation otherwise needs
access to the declared setuptools/wheel versions.

## External models and references

Six pretrained models are uploaded to [Hugging Face Models](https://huggingface.co/LeyiZhang1/PeriClock).
The [dataset repository](https://huggingface.co/datasets/LeyiZhang1/PeriClock-data) is receiving uploads; check
[resource access](../RESOURCE_DOWNLOAD.md) for per-file availability and revisions.
Both repositories are currently private and require authorized access.

```bash
python -m pip install ".[hub]"
huggingface-cli login
python resources/download.py --model PeriClock-bulk --output-dir /your/storage/models
```

The optional Hub client is only needed for SDK/authenticated downloads; prepared
local model loading does not require it. Model priority remains
`model_path > model_dir > PERICLOCK_MODEL_DIR > legacy models/`.
References and CellTypist weights use independent explicit paths.

The notebooks also accept `PERICLOCK_RESOURCE_DIR` as common external storage
(default `~/periclock-resources`); individual file paths override it. Five reviewed
aggregate Tutorial CSVs are bundled under `resources/`. Exact sizes, SHA256 and
current upload/access status are in the resource guide. The optional
`resources/download.py` helper uses verified catalog repository/revision defaults and requires an external
output directory; it never runs during package import.

## Verification

```bash
python -m pytest tests/ -q -m "not integration"
export PERICLOCK_MODEL_DIR="/path/to/downloaded/models"
python check_models.py --model-dir "/path/to/downloaded/models"
python -m pytest tests/ -q
```

See [test instructions](../tests/README.md). Optional offline installation checks:
`python -m pytest tests/installation_validation.py -q`.

Research comparators may need separate Python/R environments. In particular,
scAgeClock uses its independent environment; do not force those dependencies into
the PeriClock environment. Paper-specific environments belong to the second-phase
Tutorials preparation.
