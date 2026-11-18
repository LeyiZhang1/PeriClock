# Paper analysis Tutorials

These notebooks contain selected original research cells in Results order.
They are not a replacement analysis framework. Each notebook states its source
file and zero-based cell indices; original cell hashes are retained in cell
metadata. Only input/output paths, necessary directory creation and one explicit
variable initialization were adapted. Original imports remain in their source
cells, and calculations, model settings and tests remain unchanged.

For ordinary model use and configurable annotation resources, start with
[the package example](../examples/examples.ipynb). Research analysis requires
the original prepared inputs below; installing PeriClock does not create them.
Five approved aggregate CSVs are now bundled under `resources/`; large-resource
uploads are in progress in [Hugging Face Datasets](https://huggingface.co/datasets/LeyiZhang1/PeriClock-data); six clock
models are uploaded to [Hugging Face Models](https://huggingface.co/LeyiZhang1/PeriClock). Both repositories are
currently private. See [resource access](../RESOURCE_DOWNLOAD.md)
and [checksums/classification](../resources/catalog.json). No model, reference, manuscript, clinical input or saved notebook output is
bundled in this code checkout.

Each notebook's resource note lists only its actual read inputs and links to the
same resource guide. For individual GEO/GSM accessions and source-study choices,
consult Supplementary Table 1 of the PeriClock paper. Do not download every
original cohort simply to render a benchmark or use prepared feature matrices.
The planned first dataset payload prioritizes the full-cohort feature matrix,
the two smaller training matrices, bulk-expression table and author-confirmed full/healthy AnnData, after source
rights, privacy and file-identity clearance. Query-dependent Symphony caches
and unverified reference subsets are excluded from that first payload.

| Notebook | Paper analysis | Required original inputs | Outputs | Validation |
| --- | --- | --- | --- | --- |
| [01 Proportion clock](01_proportion_clock.ipynb) | Fig. 2a–c: 11-regressor comparison, Extra Trees evaluation and importance | Original Fig. 2 `inte_final_healthy.h5ad`; author-confirmed identity/use; source-rights/privacy review pending | Comparison CSV, fitted estimator/feature metadata, prediction and importance plots | Training not run; frozen-estimator prediction check agrees with archived metrics |
| [02 PeriClock development](02_periclock_development.ipynb) | Fig. 3b, 3e; model development underlying 3c–d and Extended Data Fig. 2 | `PeriClock-200s_features_matrix.csv`, `PeriClock-700s_features_matrix.csv`, `PeriClock-1000s_PeriClock_features_matrix.pkl`, `bootstrap_selected_stable_features.csv` | Bootstrap tables, internal validation metrics/plots, new held-out-branch estimators, final-fit plot | Static only; training not run; external predictions not regenerated |
| [03 External benchmark](03_external_benchmark.ipynb) | Fig. 3f: four-metric ranking on Sound Life / WangYF_NI | Aggregate `benchmark.csv` (17 methods) | Compact PNG/PDF and ranked CSV | **Executed in a fresh kernel**; 17 input metric rows unchanged; outputs inspected |
| [04 Feature interpretation](04_stable_feature_interpretation.ipynb) | Fig. 4a–b and expression-trend analysis underlying Extended Data Fig. 3 | `inte_final.h5ad` (byte-identical to original `bbknn_all.h5ad`), `stable_feature_coef.csv`, `features_trend_trainset.csv`; original named Enrichr libraries | Expression-trend and subtype-count plots, signed enrichment tables and KEGG heatmap | Expression-trend section executed with bundled tables; large atlas and online enrichment not run |
| [05 Cell-type-specific perturbation](05_celltype_specific_perturbation.ipynb) | Fig. 5 strategy and underlying per-feature effects in Supplementary Fig. S6 | Full-cohort feature joblib DataFrame and final PeriClock model | Four age groups' perturbations, individual results, summaries and lollipop plots | Static only; optimization not run |
| [06 Gene-level perturbation](06_gene_level_perturbation.ipynb) | Underlying optimized effects for Fig. 5e / Supplementary Fig. S7 | Same full-cohort feature table and final model | Four age groups' gene, cell-type-detail and individual-result CSVs | Static only; optimization not run; assembled paper plots not generated |
| [07 Perturbation stability](07_perturbation_stability.ipynb) | Fig. 5g and Supplementary Fig. S8b | Same feature table/model; GO BP 2023 and Reactome 2022 | Bootstrap selections/success logs, Jaccard tables, permutation results and figures | Static only; 50 bootstraps/group and 5,000 permutations not run |
| [08 DeSCOPE screening](08_descope_screening.ipynb) | Fig. 6b–d, Extended Data Fig. 9a–b: downstream knockout-age statistics | Original DeSCOPE-derived `perturbed_ages.csv`; GO BP 2023 / Reactome 2022 | Screening statistics, strong-effect genes, enrichment and RNA-processing plots | Static only; upstream DeSCOPE inference and baseline generation not reproduced |
| [09 Immunotherapy](09_immunotherapy.ipynb) | Fig. 6f–h: HNSCC baseline / NSCLC immune-age response analyses | Final model, `HNSCC_Blood_celltypist_annotated.h5ad`, `GSE306542_Symphony_annotated.h5ad` with approved clinical metadata | Original clinical prediction CSV and response plots | Static only; cohort prediction not run |
| [10 Bulk PBMC clock](10_bulk_clock.ipynb) | Fig. 6i–j and Extended Data Fig. 10 | `PeriClock-bulk_features_matrix.pkl`, `stable_feature_coef_no0.csv`, final PeriClock model, both annotated clinical cohorts and matched NSCLC raw-count AnnData | Newly fitted bulk model, predictions/statistics and gene-weight comparisons | Static only; training/clinical analysis not run |

## Configure and execute

Open the notebook's first code cell and set each input file's actual absolute
path. The corresponding environment-variable names are written beside them.
Small tables default to this checkout's `resources/`. Set `PERICLOCK_RESOURCE_DIR`
for common external storage (default `~/periclock-resources`), or override any
individual path. Model files default to `PERICLOCK_MODEL_DIR` or the resource
root's `models/`; large datasets default to its `datasets/`. No pending dataset
is automatically downloaded. Inputs can be stored in unrelated directories.
All input paths are resolved
before the notebook switches to its separate `OUTPUT_DIR`; relative output
filenames from the research code then stay beneath that directory.
Choose an output directory distinct from your input directories. Start a fresh
kernel and execute cells in order. No notebook depends on another notebook's
in-memory variables. Required intermediate tables are supplied explicitly or
generated earlier in that same notebook. In Tutorial 02, the final fit deliberately
reads the original saved stable-feature table, as the research code did; the
preceding bootstrap visualization does not replace it. Tutorial 10 rereads the
bulk model/metadata it just wrote under its own output directory.

For example, to render existing benchmark results:

```bash
# benchmark.csv defaults to the verified copy in resources/benchmark/.
# PERICLOCK_BENCHMARK_CSV can optionally override that file.
export PERICLOCK_FONT_DIR="/path/to/licensed/Arial/fonts"
export PERICLOCK_TUTORIAL_OUTPUT="/path/to/separate/benchmark-output"
jupyter nbconvert --to notebook --execute tutorials/03_external_benchmark.ipynb \
  --ExecutePreprocessor.kernel_name=scanpy \
  --output benchmark.executed.ipynb --output-dir "$PERICLOCK_TUTORIAL_OUTPUT"
```

Create the chosen output directory before using nbconvert's `--output-dir`.
Use a kernel registered to the environment below, or change the kernel name to
your equivalent environment. Missing inputs cause an explicit configuration
error. They are not replaced with simulated data. Arial is an external licensed
font resource; point `PERICLOCK_FONT_DIR` to your own font directory.

Model-use API paths remain independent: `load_model(..., model_path=...)`,
`model_dir=...` or `PERICLOCK_MODEL_DIR`. The research notebooks that inspect
serialized internals take an explicit trusted artifact path in their configuration
cell. CellTypist annotation weights and Symphony/BBKNN references are separately
configured in the package example; annotation references are not clock models.

## Environment

These selected Python notebooks use the existing Scanpy environment. The inspected
versions are Python 3.12.2, NumPy 1.26.4, pandas 2.2.2, SciPy 1.11.4,
scikit-learn 1.5.1, Scanpy 1.10.2, AnnData 0.11.1, joblib 1.4.2,
Matplotlib 3.6.3, seaborn 0.13.2, statsmodels 0.14.2, GSEApy 1.1.8,
adjustText 1.2.0 and tqdm 4.67.1. They were inspected without modifying the
environment; version inspection is not an environment recreation test.
GSEApy analyses additionally need their named external pathway libraries and
network access if not cached. No R cells are presented as Python.

For Tutorials 07–08, `PERICLOCK_GMT_DIR` selects independently obtained matching
GO BP 2023 / Reactome 2022 GMT files. The catalog records the reviewed historical
cache hashes. Original parser settings and network fallbacks are retained;
Tutorial 04's online Enrichr workflow remains unchanged.

Tutorial 03 consumes benchmark summaries. Recomputing competitor predictions
is outside it: scAgeClock uses its **own** Conda environment, siAge and
scImmuAging require their original R workflows, and IDEAL-Age requires its own
model/runtime. These dependencies are not installed into Scanpy to render a table.

## Feature counts and remaining coverage

The final predictor retains **173 selected inputs**: 169 expression features
and four cell proportions. `CD4Tcm_CD27_SESN3` has a zero fitted coefficient,
leaving **172 nonzero features** (168 expression features and four proportions).
The original expression age-trend matrix has 169 rows, including that selected
zero-coefficient expression feature. Age-trend analysis excludes proportion
features. Its original filter and plotting behavior are preserved; no rows are
added or removed to reconcile a figure-caption count. These counts do not
change the predictor's feature order or input dimension.

Not every paper panel is covered. The full atlas integration and subtype
annotation, proportion-model external validation, full-cohort/competitor external
prediction reruns, proteomics/HAGR comparisons, TF/LISA networks, longevity
reversal panels, some assembled perturbation/enrichment figures, upstream
DeSCOPE inference and Extended Data Fig. 9c–d need further source/resource
verification. They have not been replaced with placeholder notebooks. Full
large-reference Symphony/BBKNN execution also remains unverified.

Tutorial 07 explicitly initializes the previously missing global gene map by
calling the original `build_feature_gene_map` helper. Its original full-cohort
preprocessing refit is retained; equivalence to the downloaded saved transforms
has not been rechecked. Tutorial 01 retains the source's `list(set(...))` subtype
ordering, so exact training repeatability across Python hash seeds is unverified.
Original wrappers that catch individual fitting/enrichment failures are retained:
inspect their success logs before accepting an incomplete comparison.

The author confirms the complete PBMC atlas (`inte_final.h5ad`, healthy+disease)
and healthy modeling subset (`inte_final_healthy.h5ad`) as formal paper resources.
Metadata counts describe their respective stages and do not invalidate them
based on a historical 209-donor summary. The healthy file's 370 filtered modeling
rows reproduce archived proportion-model metrics with the frozen estimator,
without retraining. The released 886-tree model is confirmed identical to the
model loaded by original Fig. 2. No model/feature order or scientific method changed.

The three prepared training matrices and bulk table must accompany reproduction:
current public notebooks consume them and do not regenerate identical inputs
end to end. Source-rights/privacy review is still open. Full-atlas streaming SHA256
proves the original `bbknn_all.h5ad` and canonical `inte_final.h5ad` are identical;
Tutorial 04's configuration default uses the latter to avoid duplicate payloads.
See [the unified resource guide](../RESOURCE_DOWNLOAD.md).

Only Tutorial 03 has been executed end to end during this preparation, using
the byte-identical bundled CSV and Arial fonts with nbclient and an independent local kernel.
Its 17 metric rows were checked unchanged and its PNG, PDF and ranking CSV were
verified. This is a **plotting reproduction**, not a rerun of external predictions.
Tutorial 04's expression-trend section was also executed sequentially using the
bundled 169-row trend matrix and coefficient table and produced a valid PDF;
this did not load the atlas or execute enrichment/the entire notebook.
All ten notebooks pass format/basic Python syntax and static global-name checks;
those checks do not establish scientific reproducibility. No model training,
large-cohort inference, optimization, online enrichment or DeSCOPE inference was
performed. Keep generated clinical outputs and executed notebooks private;
the output folders and `*.executed.ipynb` are ignored by Git.
