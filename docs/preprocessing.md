# Preprocessing behavior

QC is implemented in qc.py, normalization in normalization.py, and feature
construction/saved scaling in features.py. The build_model_features adapter is
in api.py. There is no periclock.preprocessing source module in the current
package; use the documented public functions.

## Repeated normalization

`normalize_expression` performs Scanpy `normalize_total(target_sum=10000)` and
`log1p`. It records `uns['periclock_normalization']`, but currently does not stop
a repeated call. Scanpy may warn and still proceed. A marker alone cannot prove
that `.X` is unchanged since an earlier call; integer-like expression alone
cannot reliably distinguish raw data from every possible transformation.

No automatic heuristic, refusal, skip or renormalization policy was introduced.
The existing repeat-call characterization test remains unchanged. Callers must
know and document the actual `.X` state and normalize only once. A future guard
requires an explicit scientific/API decision, including marker handling and
override semantics, before changing this behavior.

## Ensembl identifiers and QC

Default QC identifies `MT-` prefixes in `var_names`; it does not automatically
read `var['gene_symbols']`. Ensembl IDs therefore yield zero matches. Automatic
conversion would change QC eligibility and potentially prediction, so it is not
performed by the package automatically.

An explicit caller-side conversion, after reviewing the dataset mapping, can
preserve both identifiers on a copy:

```python
prepared = adata.copy()
symbols = prepared.var['gene_symbols']
if symbols.isna().any() or not symbols.is_unique:
    raise ValueError('Resolve missing/duplicate gene symbols before conversion')
if 'ensembl_id' in prepared.var:
    raise ValueError('Review existing ensembl_id before replacing identifiers')
prepared.var['ensembl_id'] = prepared.var_names.astype(str)
prepared.var_names = symbols.astype(str)
```

This recipe requires the symbols already to be available and unique. It does
not infer a mapping, contact an external service, merge duplicate genes or alter
expression values. The caller's original AnnData remains unchanged. Each dataset requires its own reviewed mapping.

The two-step QC API permits a custom prefix through `calculate_qc_metrics`;
`run_qc` still does not forward that prefix. This known dispatch limitation is
documented and retains its regression test rather than being changed silently.
