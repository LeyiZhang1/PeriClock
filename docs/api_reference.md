# API reference

All examples below use the public interfaces present in `periclock` 0.1.0.dev0.
Model files are trusted external resources; functions do not download them.

## Model discovery

### `list_models()`

Returns the six supported model names as a list.

### `load_model(model_name="PeriClock", model_dir=None, model_path=None)`

`model_path` overrides `model_dir` when both are provided. Assets are downloaded
separately; all path arguments accept their actual user-selected locations.

Loads and validates a saved artifact. `model_dir` overrides
`PERICLOCK_MODEL_DIR`, which overrides the checkout-relative `models/` default.
Returns a mapping with `name`, `kind`, `model`, `artifact`, `feature_names`, and
`path`. `kind` is `single_cell`, `bulk`, or `proportion`.

```python
bundle = load_model("PeriClock-bulk", model_dir="path/to/approved/models")
```

### `get_feature_names(model_name="PeriClock", model_dir=None)`

Returns a new list of the final estimator input names in saved order.

### `get_model_info(model_name="PeriClock", model_dir=None)`

Returns a metadata mapping: name, kind, estimator class, final feature count,
resolved path, software version, and `model_version`. The latter is `None`
where an artifact has no uniform recorded model version.

## QC and normalization

Import these from submodules:

```python
from periclock.qc import calculate_qc_metrics, filter_cells, run_qc
from periclock.normalization import normalize_expression
```

### `calculate_qc_metrics(adata, mitochondrial_prefix="MT-")`

Copies an AnnData of raw counts, creates `var['mt']`, and adds Scanpy QC metrics
such as `n_genes_by_counts`, `total_counts`, and `pct_counts_mt` to `obs`.
Returns the copy. It uses `var_names`, not a separate symbol column.

### `filter_cells(adata, min_genes=None, max_genes=None, min_counts=None, max_counts=None, max_pct_mt=None)`

Copies a QC-annotated AnnData and applies only supplied inclusive thresholds.
Returns surviving cells; raises if required QC metrics are absent or all cells
are removed. With all thresholds `None`, it retains all cells.

### `run_qc(adata, **kwargs)`

Calls `calculate_qc_metrics(adata)` followed by `filter_cells(adata, **kwargs)`.
It does not forward a custom mitochondrial prefix to the first step.

### `normalize_expression(adata, target_sum=1e4, copy=True)`

Checks that `.X` is finite and nonnegative, then applies Scanpy
`normalize_total(target_sum=...)` and `log1p`. Returns a normalized AnnData and
records `uns['periclock_normalization']`. It does not detect or prevent repeat
normalization.

## Annotation

```python
from periclock import annotate_cells
```

### `annotate_cells(query, *, method, **kwargs)`

Dispatches to one of the following methods and returns an AnnData. Unified
outputs are `obs[label_key]` (default `cell_subtype_final`) and
`obs['annotation_confidence']`.

| `method` | Required extra | Key keyword arguments | Notes |
| --- | --- | --- | --- |
| `"celltypist"` | `celltypist` | `model_path=None`, `label_key`, `confidence_threshold=None`, `min_gene_overlap=0.0`, `copy=True` | Query must be log1p(CP10K); the default local model is only available in a checkout containing that asset. |
| `"bbknn_knn"` | `bbknn` | `reference`, `reference_label_key`, `label_key`, `max_per_type=4000`, `n_hvg=2000`, `n_pcs=50`, `neighbors_within_batch=10`, `k=20`, `confidence_threshold=0.5`, `random_state=0`, `copy=True` | Query/reference need compatible normalization. |
| `"symphony"` | `symphony` | `reference`, `reference_label_key`, `exclude_batches=None`, `max_per_type=1000`, `n_hvg=2500`, `n_pcs=30`, `random_state=42`, `k=15`, `label_key`, `copy=True`, `plot=False` | Runs the current original query-dependent reference-building route. |

References are caller-supplied; the package does not provide a public reference
dataset. The confidence score type differs by method and is documented in
`uns['annotation_score_type']`.

## Feature construction

### `build_model_features(adata, *, model_bundle, individual_key, celltype_key=None, max_zeros=28)`

Recommended wrapper API. `model_bundle` must be the mapping from `load_model`.
It dispatches to the appropriate model branch using its saved artifact and
feature names. It does not normalize, annotate, fit, or create a scientific
missing-value policy.

### `build_features(adata, *, model_name, individual_key, celltype_key=None, feature_names=None, model_bundle=None, max_zeros=28)`

Original compatible API. For a single-cell model, pass
`model_bundle=bundle['artifact']`. For proportion, pass
`feature_names=bundle['feature_names']`. For bulk, pass the same feature names
and the loader bundle if current missing-gene behavior is desired.

```python
bundle = load_model("PeriClock-proportion")
X = build_features(
    adata, model_name="PeriClock-proportion", individual_key="donor_id",
    celltype_key="cell_subtype_final", feature_names=bundle["feature_names"],
)
```

Both feature functions return a donor-indexed pandas DataFrame in the required
final order. See [input requirements](input_data.md) for branch-specific
assumptions.

## Prediction

### `predict_age(X, *, model_name="PeriClock", model_bundle=None, actual_age=None)`

Validates that `X` is a nonempty finite DataFrame in the exact saved order and
calls the saved estimator. `model_bundle` may be the loader wrapper; the
single-cell original artifact form remains accepted for compatibility. If it is
omitted, the named model is loaded.

`actual_age` is an optional donor-indexed pandas Series or mapping. It adds
`Actual_Age` and `Age_Diff`; it is not supplied to the estimator.

```python
import pandas as pd

actual_age = pd.Series({donor: 42.0 for donor in X.index})
result = predict_age(
    X, model_name=bundle["name"], model_bundle=bundle, actual_age=actual_age,
)
```

## Compatibility

The existing public functions and original call signatures remain supported.
`load_model("PeriClock", "/path/to/models")` is the legacy positional form;
model_path takes precedence over model_dir and PERICLOCK_MODEL_DIR. For the
single-cell branch of the original build_features interface, pass the saved
preprocessing dictionary as model_bundle (bundle["artifact"]), not the complete
loader wrapper. build_model_features accepts that wrapper and dispatches to the
existing branch-specific feature code. Core imports resolve optional analysis
packages only when those operations are used.
