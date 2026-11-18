# Example workflow

This guide describes the checked-in notebook
[`examples/examples.ipynb`](../examples/examples.ipynb). It is a configurable example derived from the historical local 10-donor workflow.
Historical outputs are cleared in the public notebook. It is not a claim that
the example data or models may be redistributed.

## Before running

Install the package with the `celltypist` extra, then open
`examples/examples.ipynb`. Configure `input_path`, `celltypist_model_path` and
`model_dir` in its first code cell using your actual external paths (or the
notebook's documented environment variables). No data file needs to live beside
the notebook. Clock models are uploaded to [Hugging Face](https://huggingface.co/LeyiZhang1/PeriClock); dataset uploads are
in progress. See [resource access](../RESOURCE_DOWNLOAD.md) for authenticated
downloads, per-file availability and revisions.

The example contains 152,472 cells. Its QC and CellTypist calls are made one
donor at a time to limit peak memory, but every cell is processed. It may take
substantial RAM and CPU time. It neither writes a derived `.h5ad` nor changes
the input, models, or reference assets.

## Stages

| Stage | Input state | PeriClock API | Output |
| --- | --- | --- | --- |
| Identifier preparation | Unique Ensembl IDs plus a reviewed, unique `var['gene_symbols']` mapping | Caller-side mapping | Original ID retained in `var['ensembl_id']`; working `var_names` are symbols. |
| QC metrics | Raw nonnegative counts | `calculate_qc_metrics` | Counts, detected genes, and mitochondrial percentage on a temporary donor copy. |
| Normalization | Raw counts | `normalize_expression(target_sum=1e4, copy=False)` | In-memory log1p(CP10K) expression and `uns['periclock_normalization']`. |
| Annotation | Log1p(CP10K), unique symbols | `annotate_cells(..., method='celltypist')` | `obs['cell_subtype_final']`, `obs['annotation_confidence']`, annotation metadata. |
| Features | Normalized, annotated AnnData | `build_model_features` | Donor × final-model-input DataFrame. |
| Prediction | Feature DataFrame in saved order | `predict_age` | `Predicted_Age`, optionally `Actual_Age` and `Age_Diff`. |

No cell filtering threshold is selected in the notebook. The current project
does not have an approved, dataset-independent filtering policy. Review QC
metrics and make a documented study-specific choice before introducing one.

## Annotation choices

The executed path uses CellTypist. The notebook also provides non-executed
templates for these alternatives:

```python
annotate_cells(
    adata, method="bbknn_knn", reference=reference,
    reference_label_key="cell_subtype_final", label_key="cell_subtype_final",
    max_per_type=1000, n_hvg=2000, n_pcs=50,
    neighbors_within_batch=10, k=20, confidence_threshold=0.5,
    random_state=42,
)
```

```python
annotate_cells(
    adata, method="symphony", reference=reference,
    reference_label_key="cell_subtype_final", label_key="cell_subtype_final",
    max_per_type=1000, n_hvg=2500, n_pcs=30, k=15, random_state=42,
)
```

BBKNN+KNN needs a labeled reference in the same expression scale as the query.
Symphony constructs a query-dependent reference internally. Both require an
approved reference with compatible labels; neither reference is distributed by
this package.

## Using an already prepared AnnData

If the query is already normalized and has compatible labels, the core
prediction route does not need Scanpy or an annotation extra:

```python
from periclock import load_model, build_model_features, predict_age

bundle = load_model("PeriClock", model_dir="/path/to/downloaded/models")
X = build_model_features(
    adata,
    model_bundle=bundle,
    individual_key="donor_id",
    celltype_key="cell_subtype_final",
)
result = predict_age(X, model_name=bundle["name"], model_bundle=bundle)
```

For `PeriClock-bulk`, omit `celltype_key`. `actual_age`, when supplied to
`predict_age`, is a donor-indexed Series or mapping used only to add comparison
columns; it is not a model input.

## Example validation status

The public notebook contains no executed donor outputs or embedded data.
It has not been rerun on the complete 10-donor dataset in this release preparation.
Paper-specific [Tutorials](../tutorials/README.md) contain selected original
analyses with separate input/validation requirements. Five aggregate inputs are
bundled; large inputs and permissions are listed in [resource access](../RESOURCE_DOWNLOAD.md).
Small synthetic smoke tests and real-model regression tests have separate scopes;
see [test instructions](../tests/README.md).

Full large-reference Symphony/BBKNN end-to-end validation remains incomplete.
