# PeriClock resource downloads

- **Models:** [LeyiZhang1/PeriClock](https://huggingface.co/LeyiZhang1/PeriClock) — all six clock models uploaded.
- **Datasets:** [LeyiZhang1/PeriClock-data](https://huggingface.co/datasets/LeyiZhang1/PeriClock-data) — repository created; uploads in progress.

Both repositories are **currently private**. Files marked available below are
available to authorized users, not anonymous public downloads. Dataset presence
was checked at commit `7eaf53c59e3b5320b5f83cde1ed2ee15ff5fa4ef` on 2026-11-18; a listed pending
file was absent from that snapshot. Upload progress may change after this check.
No ongoing upload is modified by these instructions.

File sizes and SHA256 are retained in [resources/catalog.json](resources/catalog.json).
For original GEO/GSM identifiers, study references and selected samples, consult
Supplementary Table 1. Original third-party cohorts are obtained from their sources,
not automatically mirrored in the PeriClock dataset repository.

```bash
export PERICLOCK_RESOURCE_DIR="/your/chosen/periclock-resources"
export PERICLOCK_MODEL_DIR="$PERICLOCK_RESOURCE_DIR/models"
python -m pip install ".[hub]"
huggingface-cli login  # authorized access while the repositories are private
```

The Hub client also accepts `HF_TOKEN`. Keep credentials private. Model files,
data and references can be stored independently; the package never downloads
anything during import. Large tutorial inputs default to
`$PERICLOCK_RESOURCE_DIR/datasets/`; individual path overrides remain supported.

### Checksum-verified downloads

```bash
# Offline list with filenames and current catalog status
python resources/download.py --list

# Defaults to the verified model repository and pinned commit
python resources/download.py --model PeriClock-bulk \
  --output-dir "$PERICLOCK_MODEL_DIR"

# The currently verified dataset file (downloads about 50 MB when explicitly run)
python resources/download.py --dataset bulk-training \
  --output-dir "$PERICLOCK_RESOURCE_DIR/datasets"
```

The [helper](resources/download.py) uses the authenticated Hub client automatically
for catalog resources marked private; `--use-hub` can also select it explicitly.
It verifies exact size/SHA256, protects existing files, uses temporary downloads,
and rejects storage inside the code repository. It supports one dataset at a time
or `--model all`. Pending uploads have no verified downloadable revision and are
not selected automatically. After checking a newer uploaded commit, users may
supply explicit `--repo-id` and `--revision` overrides. Do not guess commit IDs.

The underlying `huggingface_hub.hf_hub_download` interface is documented in the
[official Hub reference](https://huggingface.co/docs/huggingface_hub/package_reference/file_download).
A real download of `PeriClock-bulk.pkl` at the pinned model commit passed SHA256
and the existing loader (131 genes). No large dataset was downloaded for testing.
The clock-model, code and derived-data license decisions remain separate; uploaded
private files do not imply approval for public redistribution.

## 1. PeriClock pretrained models

Verified model commit: `0e761ebae1010f9c148e0ec7be89063815f21d25`. All six remote sizes and LFS SHA256 values match
the reviewed original artifacts. Links require authorization while private.

| Artifact / direct download | Bytes | Ordered inputs | SHA256 |
| --- | ---: | ---: | --- |
| [PeriClock.pkl](https://huggingface.co/LeyiZhang1/PeriClock/resolve/0e761ebae1010f9c148e0ec7be89063815f21d25/PeriClock.pkl) | 40,558,294 | 173 | `9c64ed8ccceb15b652df44e1a07263ca471807adc7d4d33b77db77712642962d` |
| [PeriClock-200s.pkl](https://huggingface.co/LeyiZhang1/PeriClock/resolve/0e761ebae1010f9c148e0ec7be89063815f21d25/PeriClock-200s.pkl) | 40,095,986 | 67 | `ead08a154964b634ed1628e7af03189e75a9680ef7a8e66a96a2d47c715e90ed` |
| [PeriClock-700s.pkl](https://huggingface.co/LeyiZhang1/PeriClock/resolve/0e761ebae1010f9c148e0ec7be89063815f21d25/PeriClock-700s.pkl) | 40,431,474 | 119 | `384aaeae1293fa8f7ce7bdae3fa4e4b8bfc0aa3d3d9bb77045272ce38406534f` |
| [PeriClock-1000s.pkl](https://huggingface.co/LeyiZhang1/PeriClock/resolve/0e761ebae1010f9c148e0ec7be89063815f21d25/PeriClock-1000s.pkl) | 40,591,970 | 138 | `2225e881e8cbc3d1a5f8ef462698d7e73bddccbc314f6527521eda2c782eaecc` |
| [PeriClock-bulk.pkl](https://huggingface.co/LeyiZhang1/PeriClock/resolve/0e761ebae1010f9c148e0ec7be89063815f21d25/PeriClock-bulk.pkl) | 38,228 | 131 | `271a26d99b198994a9ff624cf9b33e0c1f22ab6bbbed5819b0c14c8960f1e92e` |
| [PeriClock-proportion.pkl](https://huggingface.co/LeyiZhang1/PeriClock/resolve/0e761ebae1010f9c148e0ec7be89063815f21d25/PeriClock-proportion.pkl) | 2,830,910 | 48 | `a3a24ad811ccc17141c27087ec3ea20c7826e3c7b36e8a69f8bdf3777f497281` |

All six copied-file hashes and ordered-feature fingerprints match the frozen
regression baseline. The four single-cell artifacts match the original research
files byte for byte. Bulk/proportion use package-compatible wrappers whose
learned arrays/trees and feature order match the PeriGenes/final Extra Trees
estimators. Do not substitute historical all-gene bulk models or unwrapped
research estimators. Inspected serialization dependencies are Python 3.12,
NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.5.1 and joblib 1.4.2.

```python
from periclock import load_model
bundle = load_model("PeriClock", model_path="/your/storage/PeriClock.pkl")
bundle = load_model("PeriClock", model_dir="/your/storage/models")
```

Lookup remains `model_path > model_dir > PERICLOCK_MODEL_DIR > legacy models/`.
Only load trusted artifacts. A single prediction needs its selected model;
full integration tests/diagnostics need all six. Tutorials 05-07 and 09-10 read
the final PeriClock artifact. Tutorials 01,02,10 generate their own research
estimators; Tutorial 10 does not require downloading PeriClock-bulk.pkl to train.

### Dataset file availability

Repository: [LeyiZhang1/PeriClock-data](https://huggingface.co/datasets/LeyiZhang1/PeriClock-data). Only files confirmed in commit
`7eaf53c59e3b5320b5f83cde1ed2ee15ff5fa4ef` are marked available. Pending filenames link to no
nonexistent download; check the repository for later upload progress.

| File | Bytes | Expected SHA256 | Availability at checked commit |
| --- | ---: | --- | --- |
| `inte_final_healthy.h5ad` | 65,946,505,041 | `3fe312fc741c3777a24f940adff5eb2ab0480ed93e14934a973703ef98ad2626` | Pending upload |
| `inte_final.h5ad` | 145,696,029,571 | `c661bb9a6a4379655defe6b0ac1b792a0755021880105356251d7beeb90054b3` | Pending upload |
| `sample_10donors.h5ad` | 710,876,467 | `80e65c7304631075c732f979a913bdaa5821fc8f8ca911399c950ea249137b4b` | Pending upload |
| `inte_final_healthy_ref100k.h5ad` | 360,455,410 | `44a71aaeec2e200491d1be3a5ef660ec5678300ce9aa0941c66556048f667fb7` | Pending upload |
| `symphony_reference.h5ad` | 610,144,522 | `50d7b85d32dbd602b5d73b11b347771bafd19ff650f23e67deac964397d0c9ca` | Pending upload |
| `PeriClock-200s_features_matrix.csv` | 1,106,407,487 | `8a97e820c035c78c74b2f726826dccb68becd0ec8e969754564a8c67e8f27564` | Pending upload |
| `PeriClock-700s_features_matrix.csv` | 3,911,716,059 | `bb44f97f05b4e188d7423e4591b75e60394c777f912b957e969f76fd333ab20a` | Pending upload |
| `PeriClock-1000s_PeriClock_features_matrix.pkl` | 2,412,549,629 | `1b46ec5adc4f5350746852fe80f20c7c2403257848af1103e0bb9b78ad6470ac` | Pending upload |
| [PeriClock-bulk_features_matrix.pkl](https://huggingface.co/datasets/LeyiZhang1/PeriClock-data/resolve/7eaf53c59e3b5320b5f83cde1ed2ee15ff5fa4ef/PeriClock-bulk_features_matrix.pkl) | 50,053,770 | `fcf6ea8db66608200884872b0c5a82c6328d811558e0bd369625fd738e5983c5` | Available to authorized users |
| `perturbed_ages.csv` | 264,243,388 | `b0929f47a024185f62b9cd2e02cf3799022f077c4806f27f693ea067e983615c` | Pending upload |

## 2. Integrated PBMC atlas

`inte_final.h5ad` is the author-confirmed paper PBMC integration/annotation atlas,
containing healthy and disease data. It is distinct from both the healthy subset
and the later 1,133-row expanded training feature matrix. Proposed HF Dataset
basename: `inte_final.h5ad`; availability: **see the per-file availability table; public redistribution review remains pending**.
Tutorial 04 uses it for subtype-to-major-cell-type mapping; it is also a provenance
input for atlas annotation. Set `PERICLOCK_ANNOTATED_ATLAS` to override its path.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `inte_final.h5ad` | 145,696,029,571 | `c661bb9a6a4379655defe6b0ac1b792a0755021880105356251d7beeb90054b3` |

Checked via HDF5 metadata: **3,500,797 cells × 11,211 genes**, 395 distinct
nonmissing `Individual` values, 542 distinct Individual–Age pairs and 230 batches.
There are 2,544,543 healthy cells and 956,254 disease cells. Counts reflect the
stored identifiers at this study stage; they are not a claim of globally resolved
donor identities across all source cohorts.

Full-byte SHA256 equals the historical `conca_disease/bbknn_all.h5ad` artifact.
These are one payload with two original source paths, not two datasets. Tutorial
04 now defaults to the canonical `inte_final.h5ad` basename; its read/analysis
code is unchanged. Do not upload a duplicate `bbknn_all.h5ad`.

## 3. Healthy PBMC modeling dataset

`inte_final_healthy.h5ad` is the author-confirmed healthy modeling dataset obtained
by removing disease data from the integrated atlas. It is also the full healthy
base annotation reference. Proposed HF Dataset basename: `inte_final_healthy.h5ad`;
availability: **see the per-file availability table; public redistribution review remains pending**. Tutorial 01 reads it;
annotation can use the same cleared file without another copy. Configure
`PERICLOCK_REFERENCE_H5AD` or pass its actual path to the annotation API.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `inte_final_healthy.h5ad` | 65,946,505,041 | `3fe312fc741c3777a24f940adff5eb2ab0480ed93e14934a973703ef98ad2626` |

Checked: **2,544,543 cells × 11,211 genes**, 259 distinct nonmissing `Individual`
values, 406 Individual–Age pairs, 94 batches, and only `healthy` health-state labels.
These counts describe this dataset/stage. They do not invalidate the file based
on historical 209-donor modeling summaries. The separate 209-row feature matrix
is listed below. The copy in `PeriClock_data` has the same reviewed SHA256.

Both AnnData files have CSR float64 `X`, `log1p` metadata, unique gene indices,
1,603 retained source-suffixed `var` fields, PCA/UMAP embeddings and neighbor graphs.
Neither has `.raw` or additional `layers`. The first 10,000 stored X values checked
are finite, nonnegative and noninteger: these files must not be treated as raw UMI
counts or silently renormalized. This sample and metadata do not certify the full
matrix's numeric validity or independently prove the exact normalization history.
The full expression matrices were not loaded into memory.

Necessary `obs` fields include `Individual`, `Age`, `Age_group`, `Age_group2`,
`batch`, `health_state`, `cell_type`, `cell_subtype`, and `cell_subtype_final`, with
original QC/cluster fields retained. `var` contains mitochondrial flags and
source-suffixed gene identifiers/feature types/genome/QC fields; no fields were
removed. `uns` contains plot colors, clustering/neighbor/PCA/UMAP parameters,
`log1p`, and ranked-gene results. Inspected scalar text parameters contain no
developer absolute paths. No direct name/contact field was detected by schema
inspection; donor identifiers, cell indices, exact ages, source batch and health
state still permit linkage and require human-data review. This is not privacy
clearance or a full free-text de-identification audit.

Source redistribution review is per original study (Supplementary Table 1,
Synapse and RIKEN sources below), including disease studies represented in the
full atlas. No blanket CC BY grant is assumed. Local disk has sufficient space
(~16 TB free at preflight). Both datasets are now locally staged at the author’s
request and SHA256-verified; public redistribution checks remain open. The internal
staging map retains original source paths and verified local copies. No original
source permissions, data bytes, annotation or normalization were changed.

## 4. Training feature matrices

The current public tutorials **do not contain a complete end-to-end procedure
from AnnData to these exact three training matrices**, with all source snapshots,
annotation, joins, metadata handling and column order. Partial feature-building
code in the research archive is not an equivalent public reconstruction path.
All three matrices are therefore **required reproduction resources**. Their
verified upload status is listed above; unlisted revisions must not be assumed
available. Public source-rights/privacy clearance remains separate.

| Matrix | Bytes | Shape (rows × stored columns) | Model stage / Tutorial | Path override |
| --- | ---: | --- | --- | --- |
| `PeriClock-200s_features_matrix.csv` | 1,106,407,487 | 209 × 538,178 | Integration; PeriClock-200s; 02 | `PERICLOCK_FEATURES_200` |
| `PeriClock-700s_features_matrix.csv` | 3,911,716,059 | 712 × 528,915 | Integration+AIDA; PeriClock-700s; 02 | `PERICLOCK_FEATURES_700` |
| `PeriClock-1000s_PeriClock_features_matrix.pkl` | 2,412,549,629 | 1,133 × 528,865 | Integration+AIDA+CIMA; PeriClock-1000s and final PeriClock; 02,05-07 | `PERICLOCK_FEATURES_1000` |
| `PeriClock-bulk_features_matrix.pkl` | 50,053,770 | 1,133 × 11,018 | Original proportion-weighted bulk input; 10 | `PERICLOCK_BULK_TRAINING_TABLE` |

`PeriClock-bulk_features_matrix.pkl` is also required by the current tutorial: the original aggregation
from the full feature matrix exists in the archive but is not included in this
public notebook. Publishing it preserves the original bulk-training input rather
than silently substituting a newly defined pseudobulk.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `PeriClock-200s_features_matrix.csv` | 1,106,407,487 | `8a97e820c035c78c74b2f726826dccb68becd0ec8e969754564a8c67e8f27564` |
| `PeriClock-700s_features_matrix.csv` | 3,911,716,059 | `bb44f97f05b4e188d7423e4591b75e60394c777f912b957e969f76fd333ab20a` |
| `PeriClock-1000s_PeriClock_features_matrix.pkl` | 2,412,549,629 | `1b46ec5adc4f5350746852fe80f20c7c2403257848af1103e0bb9b78ad6470ac` |
| `PeriClock-bulk_features_matrix.pkl` | 50,053,770 | `fcf6ea8db66608200884872b0c5a82c6328d811558e0bd369625fd738e5983c5` |

CSV inputs retain `Individual` and `Age`; the two joblib DataFrames use an
`Individual` index and retain `Age`. The three single-cell matrices contain 48
subtype-ratio columns plus subtype–gene expression. The full and bulk DataFrames
are float32 with unique donor indices. Preserve original bytes, fields and order.
In particular, `PeriClock-700s_features_matrix.csv` has an original blank first header read by pandas
as `Unnamed: 0`; that name is in the saved preprocessing inputs. Do not drop it or
rewrite an apparent index column during publication.

All selected features and saved preprocessing input columns were found in their
corresponding matrices: 67/119/138/173 selected features for the four single-cell
models and 131 bulk genes. Use the saved `prop_cols`, `expr_cols`,
`selected_features`, or bulk `model_info.feature_names` for ordering; physical
matrix column order is not necessarily final estimator order. This is schema
compatibility, not refitting or full scientific reproduction. SHA256 pins the
stored version; available uploads are pinned to verified Hub commits above.
No independent scientific-processing release tag has been assigned.

Exact 209-/712-row export chains remain incompletely documented (related original
writers use different basenames). The final CIMA matrix has an explicit original
writer joining prior/CIMA features on common columns and saving float32; bulk
aggregation code also exists. These provenance limitations do not justify picking
another matrix or altering its contents. All four contain donor-level ages and
identifiers/expression and require per-source derived-data rights/privacy review.

`perturbed_ages.csv` (264,243,388 B; Tutorial 08) contains 3,445,043 donor–gene
records for 421 CIMA donors and 8,183 knockouts. Its `delta_age` is exactly the
upstream `delta_age_vs_baseline`, renamed on export: `ko_pred_age - baseline_pred_age`.
The upstream worker computes expression changes against a same-cell, zero-embedding
DeSCOPE null before adding them to observed expression and predicting clock age.
The upstream `null_pred_age` and KO-minus-null column are distinct and are not
included in this six-column CSV. Tutorial 08 tests the stored baseline-relative
differences unchanged. See the catalog for field definitions; no inference or
statistical results were regenerated. Public redistribution remains pending; remote upload status is listed above.

Five small aggregate resources already have their single formal home in GitHub:

| File / download within GitHub | Bytes | Tutorial | Availability |
| --- | ---: | --- | --- |
| [bootstrap_selected_stable_features.csv](resources/features/bootstrap_selected_stable_features.csv) | 6,398 | 02 | Included locally; publication push/license identifier pending |
| [stable_feature_coef.csv](resources/features/stable_feature_coef.csv) | 13,564 | 04 | Included locally; publication push/license identifier pending |
| [stable_feature_coef_no0.csv](resources/features/stable_feature_coef_no0.csv) | 14,247 | 10 | Included locally; publication push/license identifier pending |
| [features_trend_trainset.csv](resources/features/features_trend_trainset.csv) | 29,062 | 04 | Included locally; publication push/license identifier pending |
| [benchmark.csv](resources/benchmark/benchmark.csv) | 1,115 | 03 | Included locally; publication push/license identifier pending |

The stable/coefficient tables match all 173 ordered inputs. There are 172 nonzero
coefficients (168 expression and four proportions). The original trend matrix
contains all 169 selected expression features, including the zero-coefficient
feature, and no proportion rows. Original fields, index columns, decimals and
hashes are preserved; CSV decimals round-trip to the model's float32 coefficients.
Selection/coefficients/trends have inspected original selection, extraction and
age-group aggregation code. Exact initial assembly of benchmark.csv is not
established by its rendering cell; its 17-method values were checked against
Fig. 3f and its rendering was executed. Detailed producer evidence is in the
catalog; source-cell inspection is not a claim of rerunning every producer.

## 5. Annotation references

| Resource | Bytes | Purpose / scope | Availability |
| --- | ---: | --- | --- |
| inte_final.h5ad | See section 2 | Complete annotated PBMC atlas and cell-annotation reference | See per-file upload status; source-rights/privacy review remains pending |
| inte_final_healthy_ref100k.h5ad | 360,455,410 | Author-provided healthy reference subset, 100,000 cells × 11,211 genes, with subtype/batch labels | Optional dataset resource; exact sampling/export specification is not supplied; not a replacement modeling cohort |
| symphony_reference.h5ad | 610,144,522 | Built from the 100k subset and a construction query; 28,874 cells × 2,500 genes with standardized X, PCA and Harmony | Prepared, query-dependent Symphony reference; compatibility with new queries is not universal |
| sample_10donors.h5ad | 710,876,467 | Ten-donor usage example, 152,472 cells × 32,357 genes; Allen atlas subject/sample metadata | Example resource; source-specific subset permissions and privacy review pending |
| celltypist_model_cell_subtype_final.pkl | 1,117,390 | Compatible subtype classifier, independent of the six clock models | Separate annotation-model clearance; Pending publication |

The original analysis includes completed donor-batched BBKNN/Symphony examples for
one 486,418-cell query. These examples do not establish arbitrary-query accuracy
or equivalence between batched and whole-query annotation. Batch size must be
chosen and validated for the method and dataset; 100,000 is not a universal rule.

The Symphony cache was built by intersecting reference/query genes before HVG
selection. It retains standardized expression, PCA and Harmony state. Use prepared
reference mapping through `periclock.annotation.annotate_symphony` only with a
compatible query; do not normalize or rebuild this cache as raw expression.
`annotate_cells(..., method="symphony", reference=...)` instead constructs a new
reference from compatible base expression. Reference/model paths remain explicit:

```python
from periclock import annotate_cells
annotated = annotate_cells(adata, method="celltypist",
                          model_path="/your/storage/compatible-celltypist.pkl")
annotated = annotate_cells(adata, method="symphony",  # or "bbknn_knn"
                          reference="/your/storage/compatible-base-reference.h5ad",
                          reference_label_key="cell_subtype_final")
```

The full-reference end-to-end workflows have not been independently reproduced
for arbitrary queries. Follow the usage example for query gene identifiers and
expression preparation. No reference/model is automatically downloaded.

## 6. External validation datasets

Third-party original datasets remain official-source downloads, not PeriClock HF
mirrors. Accession/source references below were checked against manuscript Data
availability, Methods and cited studies. Individual GEO/GSM choices remain in
Supplementary Table 1. Store authorized original downloads outside the checkout;
configure exact prepared notebook inputs through the file overrides above.

| Dataset / original paper | Official accession or source | Use / Tutorial | Format and access |
| --- | --- | --- | --- |
| Integration GEO studies; Supplementary Table 1 | [GEO](https://www.ncbi.nlm.nih.gov/geo/); individual GSE/GSM identifiers in Supplementary Table 1 | Atlas/healthy reference; ancestry of 01,02,04-07,10 | Per-study formats/terms; complete study-specific rights review still needed |
| Healthy blood atlas; Terekhova et al., Immunity 2023 [17] | [syn49637038](https://www.synapse.org/Synapse:syn49637038) | Integration/reference source | Source processed/count archives; inspect Synapse file-specific access and reuse terms |
| Supercentenarians; Hashimoto et al., PNAS 2019 [18] | [RIKEN SC2018](http://gerg.gsc.riken.jp/SC2018/) | Two selected source samples in Integration | Match supplementary sample choices and original file formats; legacy link timed out during this check |
| AIDA; Kock et al., Cell 2025 [19] | [CELLxGENE ced320a1-29f3-47c1-a735-513c7084d508](https://cellxgene.cziscience.com/collections/ced320a1-29f3-47c1-a735-513c7084d508), manuscript version 1 | Expanded matrices for 02,05-07,10 | H5AD; pin original asset/version and source license; portal availability alone is not clearance |
| CIMA; Yin et al., Science 2026 [20] | [CIMA resource](https://db.cngb.org/trueblood/cima/resource); `CIMA_RNA_6484974cells_36326genes_compressed.h5ad` | Final matrices; upstream 08 | H5AD/metadata; official page provides HTTP/FTP/RaySync download instructions; derived-data rights remain unconfirmed |
| WangYF_NI; Wang et al., Nat Immunol 2025 [21] | [syn61609846](https://www.synapse.org/Synapse:syn61609846) | External validation; 03 uses aggregate benchmark only | Source `scRNA-seqProcessedLabelledObject.rds`; inspect Synapse-specific access conditions |
| Sound Life; Gong et al., Nature 2025 [22] | [Allen Human Immune Health Atlas](https://apps.allenimmunology.org/aifi/resources/imm-health-atlas/downloads/scrna/); `human_immune_health_atlas_full.h5ad` | External validation; 03 uses aggregate benchmark only | H5AD with subject/sample metadata; source terms apply; current portal notes a 2025-06-10 reprocessing/privacy update, so match the research snapshot |

Tutorial 03 needs none of these large cohorts just to render its bundled rankings.
Official raw/source data are not automatically byte-identical to prepared annotated
inputs. Synapse landing records, CIMA and Allen resource pages were reachable;
full current license/access conditions were not established. Other source pages
may require JavaScript/authentication or did not respond in this environment.

## 7. Immunotherapy datasets

| Dataset / original paper | Official accession or download route | Actual tutorial input | Access / publication |
| --- | --- | --- | --- |
| HNSCC; Liu et al., Nat Med 2025 [50] | [PRJCA027702](https://ngdc.cncb.ac.cn/bioproject/browse/PRJCA027702), [PRJCA028161](https://ngdc.cncb.ac.cn/bioproject/browse/PRJCA028161); [study](https://www.nature.com/articles/s41591-025-03873-7) | 09-10: `HNSCC_Blood_celltypist_annotated.h5ad` (2,430,848,758 B), clinical metadata | Manuscript says processed data were author-provided. Request access through the original authors; no clinical payload staged |
| NSCLC; Figueiredo et al., iScience 2026 [51] | [GEO GSE306542](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE306542) | 09-10: `GSE306542_Symphony_annotated.h5ad` (181,915,477 B); 10 also needs exact matched `GSE306542_processed.h5ad` raw counts | Follow GEO/study access terms. Prepared clinical-file redistribution is not cleared; exact raw-count AnnData not found in the four resource roots |

These accessions are manuscript-confirmed; GEO/BioProject records could not be
retrieved in this environment, so current file-level access was not verified.
Sequence availability does not grant redistribution of processed clinical files.
Set `PERICLOCK_HNSCC_H5AD`, `PERICLOCK_NSCLC_H5AD` and
`PERICLOCK_NSCLC_COUNTS` to authorized prepared inputs. Do not infer clinical
values or substitute normalized data for the missing raw-count object.

## 8. Other reference databases

| Source / original reference | Official acquisition route | Paper use / current Tutorial dependency | Handling |
| --- | --- | --- | --- |
| Enrichr GO BP/CC/MF 2025, KEGG 2021 | [Enrichr libraries](https://maayanlab.cloud/Enrichr/#libraries) | 04 retains original online signed enrichment | No payload mirror; historical service snapshot unpinned |
| GO BP 2023 / Reactome 2022 | [Enrichr libraries](https://maayanlab.cloud/Enrichr/#libraries) | 07-08; exact local cache SHA256 in catalog | Independently obtain matching GMTs in PERICLOCK_GMT_DIR; original parser settings/network fallback remain; same name alone is insufficient |
| GenAge / CellAge [32,33] | [GenAge](https://genomics.senescence.info/genes/) / [CellAge](https://genomics.senescence.info/cells/) | Paper Fig. 4e; not read by current ten notebooks | Obtain source gene tables under their terms; exact historical release unpinned; no mirror |
| MSigDB Hallmark / GO-BP | Original library distributions / Enrichr as used in original analysis | Paper longevity analysis; not a direct input of current ten notebooks | Match the original release; do not add unused GMT downloads to notebooks |
| LISA hg38 and sciPro regulatory-potential reference [36] | Original cited tool/data distributions; exact RP snapshot/link not established here | Paper TF/regulatory analysis; not a current ten-notebook input | No guessed download link and no unverified RP payload staged |
| DeSCOPE checkpoint / ESM-2 embeddings [16] | [Cited DeSCOPE work](https://doi.org/10.64898/2026.04.13.718147); exact inference checkpoint/embedding releases not pinned | Upstream virtual screen; 08 begins from the age table | Not a required download for downstream 08; no unverified model/embedding payload staged |
| Lehallier lifespan proteomic results [7] | [Original Nat Med study and supplements](https://www.nature.com/articles/s41591-019-0673-2) | Paper Fig. 4c; not read by current ten notebooks | Original association/summary tables; PeriClock manuscript supplies no separate accession; no participant-level mirror |
| Coenen cross-cohort proteomic results [31] | [Original Front Aging article and supplements](https://www.frontiersin.org/journals/aging/articles/10.3389/fragi.2023.1112109/full) | Paper Fig. 4d; not read by current ten notebooks | Original published summaries; underlying cohort access remains source-specific; no new accession inferred |
| Arial fonts | User's licensed font installation | 01-10 | Set PERICLOCK_FONT_DIR; no fonts redistributed |
