# Online Shoppers notebook

Open `notebooks/01_binary_classification.ipynb` and run all cells, or execute it from a
fresh kernel using the project's Python environment:

```bash
.venv/bin/python scripts/execute_notebook.py notebooks/01_binary_classification.ipynb --device cuda
```

Install `requirements.txt` first. The execution helper ensures that the fresh kernel
uses the same Python interpreter and writes notebook outputs in place. It may require
permission to open local kernel ports and access the GPU in restricted environments.

The first run downloads the small public UCI archive. Pre-cache the pinned classification
checkpoints before the full notebook run:

```bash
.venv/bin/python scripts/smoke_models.py --models tabpfn mitra --tasks binary --device cpu --download
```

The notebook defaults to cached model weights (`DOWNLOAD_WEIGHTS=False`); that setting
can be changed explicitly in its configuration cell. See the existing
[model access instructions](model-runtime.md) when checkpoint access is required.

## Data and protocol

The [UCI dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
is credited to Sakar and Kastro (2018), [DOI 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q),
under CC BY 4.0. The manifest in `datasets/online-shoppers.json` pins the download URL,
archive and CSV SHA-256 hashes, schema, target, source row count, and positive count.
Downloaded data remain in the ignored `.cache/datasets/` directory.

The loader checks the complete released dataset before extracting the 17 predictors and
the boolean `Revenue` target. Numeric category identifiers are treated as categorical
values. All 12,330 rows are retained in the full profile, including 125 identical full
rows; no identifiers are available to investigate those observations further.

The single stratified split uses seed 42: 7,398 training rows, 2,466 validation rows,
and 2,466 test rows. Saved row positions reconstruct the split exactly. Exploration uses
training features; model selection and all learned preprocessing use training and
validation only. Models are fitted on training rows, without a train-plus-validation
refit. No threshold tuning, balancing/resampling, or automatic row subsampling occurs.

## Model settings

The notebook compares seven predeclared rows:

- A dummy training-prior classifier.
- XGBoost and LightGBM on engineered features, each with its existing four-candidate,
  80-tree validation search.
- Frozen TabPFN-3.5-Fast and Mitra v2 on raw features.
- The same frozen foundation models on engineered features, labeled separately.

Feature engineering keeps the original predictors and adds total pages, total duration,
seconds per page, product-page fraction, and exit-rate minus bounce-rate. These are
label-free functions of each row; zero-page ratios become zero. Imputation, encodings,
scaling, and model-specific transforms fit training rows only.

Each independent recipe runs in a fresh spawned process, using the same source rows
and split positions. Worker exit releases GPU allocations. Recipes and search spaces are
fixed before the loop; earlier test scores do not influence later configurations.
Each classical winner is selected by validation ROC-AUC before its final test evaluation.
Foundation models do not use validation tuning or fine-tuning.

Mitra uses the upstream inference precision policy: bfloat16 autocast on supported
CUDA devices, float16 autocast on other CUDA devices, and float32 on CPU. Checkpoint
parameters remain frozen. The actual device and inference precision are saved with
each result; GPU precision can cause small numerical differences from CPU results.

The default budget is 300 seconds per model variant, including loading and prediction;
change it with `--model-timeout`. A missing prerequisite, known memory limit, or timeout
produces **Not evaluated** with a reason and no metric values. Unexpected failures are
marked **Failed**, saved for diagnosis, and cause the notebook's final assertion to fail.
The full TabPFN training partition exceeds its default CPU limit; CUDA is appropriate
for full-data runs. Hardware and runtime limits are properties of this run, not evidence
about a model's predictive quality.

## Outputs and interpretation

The executed notebook displays the final test table, ROC-AUC plot, average-precision
plot, and measured observations. Full artifacts live under `results/online-shoppers/full/`:

- `results.json`: source metadata, source-row/split positions, selected settings,
  preprocessing recipes, validation scores, test predictions, metrics, versions, and
  evaluated foundation-model checkpoint/device details.
- `splits.json`: explicit source rows and reproducible train/validation/test positions.
- `metrics.csv`: all model rows with metrics, status, elapsed seconds, and limitations.
- `roc_auc.png` and `average_precision.png`: plots from measured metrics only.

ROC-AUC is primary. Average precision, F1, precision, recall, and accuracy accompany it;
average precision is distinguished from trapezoidal PR-curve area. Empty CSV metrics
and the notebook's `Not evaluated` cells are not zero scores. Unavailable models are
excluded from plots, while their reasons remain in the table.

This is a retrospective single-split comparison of released session features, including
`PageValues`; availability at a specific live prediction time is not verified. Identical
rows and absent session identifiers limit independence checks. Conclusions apply only
to these observed splits/settings and do not imply statistical significance or universal
model superiority.

To regenerate a plot from saved results without fitting models:

```python
from src.benchmark import load_results
from src.evaluation import comparison_table
from src.plotting import plot_comparison

results = load_results("results/online-shoppers/full/results.json")
comparison_table(results)
plot_comparison(results).savefig("results/online-shoppers/full/roc_auc.png", dpi=150)
```

## Fast execution check

```bash
.venv/bin/python scripts/execute_notebook.py notebooks/01_binary_classification.ipynb --profile smoke --device cpu --output /tmp/online-shoppers-smoke.ipynb
```

The smoke profile uses one fixed, stratified 500-row sample with all model recipes and
writes to `results/online-shoppers/smoke/`. Source rows and the sample profile are saved
explicitly. These are execution-check scores and must not be mixed with full-data results.

Focused tests verify the cached source/checksum, schema and target exclusion, split
counts, categorical handling, row-local features, bounded workers, and honest failure
reporting:

```bash
.venv/bin/python -m unittest discover -s tests -p test_online_shoppers.py -v
```
