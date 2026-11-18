# Models and prediction outputs

PeriClock supports six separately downloaded, pretrained artifacts. Loading an
artifact validates its final input count and feature order against the saved
estimator; the package never fits or retrains a model.

| Name | Estimator input | Final input count |
| --- | --- | ---: |
| `PeriClock` | Saved-scaled cell-type expression and proportion features | 173 |
| `PeriClock-200s` | Saved-scaled cell-type expression and proportion features | 67 |
| `PeriClock-700s` | Saved-scaled cell-type expression and proportion features | 119 |
| `PeriClock-1000s` | Saved-scaled cell-type expression and proportion features | 138 |
| `PeriClock-bulk` | 131 gene columns in saved order; saved Pipeline performs scaling | 131 |
| `PeriClock-proportion` | 48 donor cell-type proportions in saved order | 48 |

The model-name suffixes are names, not a claim about the final input count.
For the four single-cell artifacts, saved VarianceThreshold, SelectKBest, and
StandardScaler objects are applied without fitting. For bulk, do not bypass the
saved Pipeline's scaler.

## Selected inputs and nonzero coefficients

The final PeriClock model was fitted on 1,133 donors. Its saved artifact retains
173 selected inputs: 169 cell-subtype-specific expression features and four
cell-proportion features. One expression feature, `CD4Tcm_CD27_SESN3`, has a fitted
coefficient of zero. The resulting 172 nonzero features comprise 168 expression
features and four proportions. The zero-coefficient input remains in the saved
feature order and must not be removed when constructing model inputs.

Expression-age trends are calculated for the 169 selected expression features,
including the zero-coefficient feature; cell-proportion features are excluded
from that analysis. PeriClock-bulk uses donor mean expression of the 131 distinct
genes represented by PeriClock, without cell-subtype resolution.

## External resources

The six clock artifacts are uploaded to [Hugging Face Models](https://huggingface.co/LeyiZhang1/PeriClock).
The [dataset repository](https://huggingface.co/datasets/LeyiZhang1/PeriClock-data) is still receiving uploads. Both repositories
are currently private; authorized access is required. Store resources wherever you
choose, outside the code checkout. See [resource access](../RESOURCE_DOWNLOAD.md)
for exact files, SHA256, revisions and the authenticated download workflow.

```python
from periclock import load_model

bundle = load_model("PeriClock", model_path="/path/to/downloaded/PeriClock.pkl")
bundle = load_model("PeriClock", model_dir="/path/to/downloaded/models")
```

```bash
export PERICLOCK_MODEL_DIR="/path/to/downloaded/models"
```

Resolution order is `model_path` > `model_dir` > `PERICLOCK_MODEL_DIR` >
legacy repository-relative `models/`. If both path arguments are provided,
`model_path` wins. The legacy default contains no models in a code-only checkout.
`load_model("PeriClock", "/path/to/models")` remains supported.
Only load trusted pickle artifacts.

## Inspecting a model

```python
from periclock import get_feature_names, get_model_info, load_model

info = get_model_info("PeriClock-bulk", model_dir="/path/to/downloaded/models")
features = get_feature_names("PeriClock-bulk", model_dir="/path/to/downloaded/models")
bundle = load_model("PeriClock-bulk", model_path="/path/to/downloaded/PeriClock-bulk.pkl")
```

`get_feature_names` returns a new list in the exact estimator order. Do not
sort, deduplicate, or manually reorder it. `model_version` is `None` because a
uniform model-release version was not recorded in all artifacts; the Python
software version is not a model version.

## Outputs

`predict_age` always returns a donor-indexed DataFrame with:

- `Predicted_Age`: the direct saved-estimator output.
- `Actual_Age`: only when `actual_age` is supplied.
- `Age_Diff`: `Predicted_Age - Actual_Age`, only when `actual_age` is supplied.

No clipping, calibration, residualization, or extra output transform is applied
by `predict_age`. These values should not be used for clinical decisions.

## Asset boundary

Model artifacts are deliberately excluded from wheels and source archives. Use
only trusted, approved copies and configure `model_dir` or
`PERICLOCK_MODEL_DIR` after a non-editable installation. Model redistribution
rights and version provenance require rights-holder confirmation.
