# Troubleshooting

## `Model not found`

Wheel and source distributions contain code only. Point to trusted external
artifacts explicitly:

```python
bundle = load_model("PeriClock", model_dir="path/to/approved/models")
```

or pass `model_path="/path/to/PeriClock.pkl"`, or set `PERICLOCK_MODEL_DIR` before importing/loading. Do not download or
unpickle untrusted model files.

## Missing annotation references or CellTypist model

Download the compatible resource separately (public links pending), then pass
`annotate_cells(adata, method="celltypist", model_path="/path/to/model.pkl")`
or `annotate_cells(adata, method="symphony", reference="/path/to/reference.h5ad")`.
BBKNN also accepts an actual `.h5ad` path via `reference`. Existing compatible
AnnData objects are accepted. Check file existence, labels and normalization;
no developer server directory is required. Full large-reference Symphony/BBKNN
validation remains incomplete.

## Tests and model diagnostics

From the repository root, in an environment with `.[dev,preprocessing]`:

```bash
# No downloaded models or reference data needed:
python -m pytest tests/ -q -m "not integration"
# Also reports skips for unconfigured model integrations:
python -m pytest tests/ -q

# After downloading all six trusted models:
export PERICLOCK_MODEL_DIR="/path/to/downloaded/models"
python -m pytest tests/ -q
python -m pytest tests/ -q -m integration
python check_models.py --model-dir "/path/to/downloaded/models"
# Or use PERICLOCK_MODEL_DIR directly:
python check_models.py
```

No environment variable: model integrations skip with an explicit reason.
An explicitly configured incomplete directory fails. Corrupt artifacts,
fingerprint differences, numerical mismatches and prediction failures fail;
none are converted to skips. The diagnostic exits nonzero if resources are
missing. These tests do not run large-reference annotation end to end.

## Optional dependency error

Install only the needed extra from the checkout root:

```bash
python -m pip install -e '.[preprocessing]'
python -m pip install -e '.[celltypist]'
python -m pip install -e '.[bbknn]'
python -m pip install -e '.[symphony]'
```

Core model loading and prediction do not require annotation packages. A broken
transitive dependency is reported as its original import error rather than
being relabeled as an absent optional package.

## Mitochondrial QC is zero with Ensembl IDs

The default QC check uses `var_names.str.startswith('MT-')`; it does not read a
symbol column automatically. Preserve original IDs and explicitly use a
reviewed, unique symbol mapping before symbol-based QC. See
[input requirements](input_data.md).

## A donor is absent from non-bulk output

The current non-bulk feature paths apply the fixed 48-type coverage gate.
Inspect compatible annotation labels and donor coverage before changing
`max_zeros`. The historical authority for this policy remains unresolved; do
not invent a label mapping to force a donor through.

## `Feature columns are not in the original training order`

Use the DataFrame returned by `build_model_features`, or use
`get_feature_names` exactly as returned. Do not alphabetize features or use the
bulk artifact's coefficient-selected subset in place of its 131 input genes.

## Repeated normalization

`normalize_expression` does not currently prevent a second transformation.
Record your input state and call it once. A marker in `.uns` is not proof that
an arbitrary AnnData is raw counts.

## Annotation consumes too much memory

Use an approved smaller query or a documented chunking strategy, and keep
query/reference preprocessing compatible. The provided example chunks by donor
for QC and CellTypist. BBKNN and Symphony need an explicit reference and can be
substantially more demanding.

## Installation or serialization mismatch

Use the Python 3.12 and pinned dependency versions in `pyproject.toml`. The
tested environment is documented in [installation](installation.md); other
Python/scikit-learn versions have not been certified for these artifacts.

## Writable runtime caches

If Scanpy import raises `cannot cache function ... no locator available`, choose
writable cache locations before starting Python:

```bash
export NUMBA_CACHE_DIR="/path/to/writable/cache/numba"
export MPLCONFIGDIR="/path/to/writable/cache/matplotlib"
```

This changes cache storage, not scientific calculations or installed packages.
