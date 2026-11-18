# Tests

Install the project with `.[dev,preprocessing]` in the model-compatible Python 3.12
environment. From the repository root:

```bash
# No external models required:
python -m pytest tests/ -q -m "not integration"
# Includes explicit skips when models are unconfigured:
python -m pytest tests/ -q
# Full suite with all six downloaded, trusted models:
export PERICLOCK_MODEL_DIR="/path/to/downloaded/models"
python -m pytest tests/ -q
python check_models.py --model-dir "/path/to/downloaded/models"
```

An unset model directory skips model integrations. A configured incomplete
folder, corrupt artifact, changed fingerprint or prediction mismatch fails.
No estimator is fitted or model overwritten.

`baselines/artifacts.json` is required by model tests: it freezes model and ordered
feature fingerprints. Do not regenerate it to accommodate failures.
`reference.py` contains independent test calculations. Existing prediction checks
retain `rtol=0, atol=1e-12`; normalization checks compare with Scanpy.

## Installation and local evidence

`python -m pytest tests/installation_validation.py -q` performs explicit offline
editable/wheel/source-distribution checks with existing dependencies. It is not a
clean network resolver test and is outside default test-file discovery.

`local_validation.py` and `baselines/local_evidence.json` are preserved local
checks tied to original private research files, including a separately provisioned
example dataset. They are excluded from Git and default discovery, not deleted or
weakened. Missing local evidence fails when these checks are explicitly invoked.
The standard unit/model suites do not depend on them.

`test_resource_download.py` checks the explicit downloader offline: exact bytes,
SHA256 rejection/cleanup, preserving existing files and refusing model storage
inside the code checkout. Its temporary download directory must be outside the
checkout (pytest's normal default is suitable). No live model download is made.

These tests do not establish paper reproduction or full large-reference
Symphony/BBKNN annotation validation.
