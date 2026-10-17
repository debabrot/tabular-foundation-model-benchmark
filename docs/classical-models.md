# XGBoost and LightGBM comparisons

Run the offline synthetic demonstration:

```bash
.venv/bin/python scripts/demo_baseline.py --classical
```

This compares XGBoost and LightGBM with the existing dummy baselines on binary,
multiclass, and regression fixtures. It saves measured metrics, predictions, tuning
traces, CSV tables, and PNG plots under `results/classical-demo/`. These are synthetic
examples, not scores for Online Shoppers, Seoul Bike, or Dry Bean.

Recreate tables and plots from saved artifacts without fitting or tuning again:

```bash
.venv/bin/python scripts/demo_baseline.py --classical --render-only
```

## Use in a notebook

```python
from src.benchmark import run_comparison, save_results
from src.data import make_split
from src.evaluation import comparison_table
from src.models.classical import classical_search_spaces, make_classical_models
from src.plotting import plot_comparison

# X and y are already loaded and aligned; task is binary, multiclass, or regression.
splits = make_split(y, task, seed=42)
results = run_comparison(
    X, y, make_classical_models(task, seed=42),
    task=task, dataset_name="Your dataset", seed=42, splits=splits,
    search_spaces=classical_search_spaces(),
)
save_results(results, "results/classical-comparison.json")
comparison_table(results)
plot_comparison(results)
```

Reuse the same saved split positions when integrating other model families. For a
chosen chronological regression protocol, supply its positions through `splits=`.

## Fixed search budget

| Family | Candidates | Trees per candidate | Fixed settings |
| --- | --- | ---: | --- |
| XGBoost | Learning rate 0.05 or 0.1 × depth 3 or 6 | 80 | CPU histogram trees, one thread, seed 42 by default, full row/column sampling |
| LightGBM | Learning rate 0.05 or 0.1 × leaf count 7 or 15 | 80 | CPU, one thread, seed 42 by default, deterministic mode, column-wise histograms, minimum 10 rows per leaf |

Each family receives four candidate fits per task, with no early stopping or extra
refit after selection. This is a candidate-count budget, not an equal wall-clock budget.
The caller can supply different finite `search_spaces`, with all candidates recorded.
The default grids are small initial baselines, not exhaustive dataset-specific tuning.

Every candidate clones and fits its own preprocessing pipeline on training rows only.
It is scored on validation data using ROC-AUC for binary classification, macro-F1 for
multiclass classification, or RMSE for regression. The first candidate wins ties.
All models finish selection before final test predictions are made. Neither test
features nor test labels are supplied to fitting or candidate scoring. The retained
winner receives final evaluation using the same train-only fitted pipeline.
No thresholds are tuned, and validation rows are not added to training.

XGBoost classification uses a training-only label encoder because its sklearn wrapper
expects contiguous integer labels. Predictions are decoded to the original labels;
probability column i maps to the saved `classes[i]`. The effective underlying XGBoost
configuration is saved alongside the adapter's parameters. LightGBM handles original
class labels directly. See the [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)
and [LightGBM determinism settings](https://lightgbm.readthedocs.io/en/stable/Parameters.html#deterministic).

## Saved evidence and checks

Each tuned model's JSON entry includes the grid, selection metric and direction,
candidate count, every candidate's validation metrics, selected index, selected model
parameters, and elapsed fit-plus-validation seconds per candidate. Timing varies between
runs; settings, split positions, predictions, and metrics reproduce in the same
environment. Installed XGBoost and LightGBM versions are included in provenance.
The shared JSON, table, and plotting interfaces remain the same as the
[baseline workflow](baseline-workflow.md).

The test suite exercises actual models on all three task types, arbitrary label
mapping, selection direction and ties, repeated-process reproducibility excluding
timing, saved table/plot regeneration, and test-set isolation. An instrumented estimator
also verifies that every candidate fits training rows only, that all selection finishes
before any test query, and that changing test targets cannot change the selected settings.

```bash
.venv/bin/python -m unittest discover -s tests -v
```
