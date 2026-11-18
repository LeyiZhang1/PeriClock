# Input data requirements

PeriClock operates on `anndata.AnnData` objects. The package does not download,
convert, or infer a reference dataset.

## General requirements

- `adata` must contain at least one cell and one gene.
- `obs_names` and `var_names` must be unique.
- The requested donor column must be present in `adata.obs` and contain no
  missing values. IDs must remain distinct after conversion to strings.
- A cell-type column is required for the four single-cell models and for
  `PeriClock-proportion`; it is not required for `PeriClock-bulk`.
- The final feature DataFrame passed to `predict_age` must be finite numeric
  values in the saved feature order. Feature construction does not normalize
  `.X` itself.

## Gene identifiers and expression state

Clock and CellTypist inputs are matched literally against `adata.var_names`.
The shipped example has Ensembl IDs in `var_names` and reviewed symbols in
`var['gene_symbols']`; it preserves Ensembl IDs in `var['ensembl_id']` and uses
the unique symbols as working names. Do not silently aggregate duplicated
symbols or guess an ID mapping. See [preprocessing status](preprocessing.md).

For `calculate_qc_metrics` and `normalize_expression`, supply raw,
nonnegative counts. The default QC mitochondrial prefix is `MT-`. Annotation
expects log1p(CP10K) expression unless an explicitly compatible reference/model
uses a different scale. Normalization must be performed once by the caller;
the current implementation records a marker but does not reject repeat calls.

## Model-specific inputs

| Model family | Required keys | Feature source | Current missing-input behavior |
| --- | --- | --- | --- |
| `PeriClock`, `PeriClock-200s`, `PeriClock-700s`, `PeriClock-1000s` | donor + compatible cell type | Donor cell-type proportions and cell-type mean normalized expression; saved selectors/scalers | Missing expression genes and absent donor/type combinations are represented as zero before saved transforms. |
| `PeriClock-proportion` | donor + compatible cell type | Donor proportions for 48 saved labels | Missing labels are zero after reindexing. |
| `PeriClock-bulk` | donor | Donor mean normalized expression for the 131 saved genes | With a loader bundle, the current release can fill absent genes with the saved scaler means; see the scientific review before interpreting this behavior. |

For the non-bulk paths, the current release retains donors with no more than 28
zero values across the fixed 48 saved cell types (at least 20 observed types).
This is a frozen implementation behavior, not a settled historical scientific
policy. Obtain a project-owner decision before using it to compare cohorts.

## Optional chronological age

Prediction does not require chronological age. To obtain `Actual_Age` and
`Age_Diff`, provide a Series or mapping indexed by the same donor identifiers:

```python
ages = adata.obs.groupby("donor_id", observed=True)["age"].first()
result = predict_age(X, model_name="PeriClock", actual_age=ages)
```

Callers are responsible for verifying that donor ages are complete, numeric,
and consistent across cells. The historical handling of repeated visits and
multiple ages remains subject to scientific review.
