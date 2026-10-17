# Shared baseline workflow

Run the offline synthetic demonstration from the repository root:

```bash
.venv/bin/python scripts/demo_baseline.py
```

This runs two dummy models for each of binary classification, multiclass classification,
and regression. Each fixture has 150 rows, four numeric features, one categorical feature,
and missing values. Binary labels are imbalanced; multiclass labels are strings.
These are synthetic demonstrations, not results for the three selected public datasets.

The command prints test-metric tables and writes one JSON artifact, CSV table, and PNG
comparison plot per task under `results/baseline-demo/`. To regenerate the CSVs and plots
from saved predictions and metrics without fitting models again:

```bash
.venv/bin/python scripts/demo_baseline.py --render-only
```

Use `--output-dir <directory>` for another artifact directory, or `--seed <integer>`
for a new deterministic demonstration. Rerunning with the same data, seed, settings, and
environment reproduces the JSON artifacts. Package versions are recorded in each file.

Add `--classical` to include validation-tuned XGBoost and LightGBM and save outputs in
`results/classical-demo/`. See [classical-model tuning](classical-models.md) for its
four-candidate budgets and recorded timings.

Add `--tabpfn` for frozen TabPFN comparisons. Its native preprocessing and checkpoint
provenance are described in [TabPFN integration](tabpfn.md). Model-specific
`preprocessors=` overrides allow native and classical preprocessing in the same run.

Add `--mitra` for frozen Mitra v2 classification and distributional regression; see
[Mitra integration](mitra.md). Combining `--classical --tabpfn --mitra` compares all
four core models on the same synthetic split.

## Use in a notebook

From a Python session with the repository root on its import path:

```python
from sklearn.dummy import DummyClassifier
from src.benchmark import load_results, run_comparison, save_results
from src.evaluation import comparison_table
from src.plotting import plot_comparison

# X: a feature DataFrame; y: an aligned Series with the target excluded from X.
results = run_comparison(
    X, y, {"Dummy prior": DummyClassifier(strategy="prior")},
    task="binary", dataset_name="Online Shoppers", positive_label=True,
)
save_results(results, "results/online-shoppers-baseline.json")
comparison_table(results)  # Test metrics; no rounding in the saved data.
plot_comparison(results)   # Displays the task's primary test metric.

# Later models can reuse the exact positional split from this artifact.
saved = load_results("results/online-shoppers-baseline.json")
split_positions = saved["splits"]
```

## Evaluation protocol

- One fixed 60/20/20 random split with seed 42 by default, stratified for classification.
  For row counts that do not divide evenly, scikit-learn's split rounding applies.
- Split indices are **row positions**, used with `iloc`, not DataFrame index labels.
  All models in a comparison receive the same partitions. Classification partitions
  must contain every class; targets and supplied split indices are validated.
- Supplied `splits=` may come from a saved result or an explicitly chosen chronological
  split. They must be exhaustive, nonoverlapping partitions; provided splits are not
  silently reshuffled. The [Seoul Bike notebook](seoul-bike.md) uses the confirmed
  random-split retrospective prediction protocol with observed weather.
- Every estimator and preprocessing pipeline is cloned and fitted on training rows only.
  Numeric features use median imputation and standard scaling. Categorical features
  normalize missing values, impute the training mode, and one-hot encode, ignoring
  unseen categories. Empty training columns are retained. Dates need explicit features.
- A model's unset `random_state` is filled with the comparison seed when that parameter
  exists; explicitly supplied seeds are preserved and recorded. Other sources of model
  randomness and device-specific behavior remain the model integration's responsibility.
- With no `search_spaces`, the workflow evaluates already-selected estimators on
  validation and final test data. Optional finite parameter grids select candidates by
  validation metrics, with all selections completed before test evaluation. It performs
  no threshold selection or train-plus-validation refit.
  Models declaring native preprocessing use their original feature inputs by default;
  explicit per-model or global preprocessing choices override that behavior.

| Task | Saved metrics | Primary plot |
| --- | --- | --- |
| Binary | ROC-AUC, average precision, F1, precision, recall, accuracy | ROC-AUC, higher is better |
| Multiclass | Macro-F1, weighted-F1, accuracy | Macro-F1, higher is better |
| Regression | RMSE, MAE, R² | RMSE, lower is better |

Binary probability scores use the column matching `positive_label` in the model's
`classes_`, even when that class appears first. If omitted, the positive label is the
last sorted training class; set it explicitly when the business meaning matters.
Average precision is reported under that name, not as trapezoidal PR-curve area.
Undefined classification precision/recall/F1 use zero; an invalid ROC-AUC partition
raises an error. Invalid predictions and nonnormalized probabilities are rejected.

## Saved evidence

JSON artifacts include the task, seed, split positions, dataset name/kind, ordered-data
fingerprint, model and preprocessing configuration, package versions, class order,
positive label, validation/test targets, predictions, probabilities, and measured metrics.
The fingerprint detects changed ordered values or schema; it does not replace the raw
dataset download checksum or source/license information in a dataset notebook.

`comparison_table()` and `plot_comparison()` accept loaded artifacts directly.
`comparison_table(results, partition="validation")` selects validation metrics explicitly.
Plotting returns a Matplotlib figure so notebooks can display it and scripts can export
and close it. Plot titles identify synthetic data; axes state whether higher or lower
values are preferable. Learned preprocessing and fitted models are not serialized.

Verify the shared workflow and existing runtime checks:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
