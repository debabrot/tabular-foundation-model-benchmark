# Frozen TabPFN comparisons

Run the shared workflow with the pinned TabPFN-3.5-Fast checkpoint and dummy baselines:

```bash
.venv/bin/python scripts/demo_baseline.py --tabpfn --device cpu
```

This writes JSON results, CSV metrics, and PNG plots for three synthetic tasks under
`results/tabpfn-demo/`. It is an integration demonstration, not public-dataset evaluation.
The default ensemble setting is `auto`: the checkpoint's recommended ensemble is used
and the resolved count is recorded. The recorded tiny-fixture run uses four estimators.
For a faster probe, explicitly pass `--tabpfn-estimators 1`; that override is saved too.

The default is offline. Add `--download` only when pinned weights are missing. Review
the [model card and access requirements](https://huggingface.co/Prior-Labs/tabpfn_3_5)
if access is needed; the integration does not perform interactive login or accept terms.
Checkpoint resolution is shared with the existing runtime smoke runner, including the
filename-preservation fix for Hub symlinks.

Recreate outputs from saved artifacts without model loading or fitting:

```bash
.venv/bin/python scripts/demo_baseline.py --tabpfn --render-only
```

To compare the classical models and TabPFN on identical rows in one run:

```bash
.venv/bin/python scripts/demo_baseline.py --classical --tabpfn --device cpu
```

Combined artifacts go to `results/combined-demo/`; use `--output-dir` to choose another
directory. The classical families retain their validation-only searches, while TabPFN
uses frozen inference without validation tuning.

## Use in notebooks

```python
from src.benchmark import run_comparison, save_results
from src.data import make_split
from src.models.tabpfn import make_tabpfn_models
from src.evaluation import comparison_table
from src.plotting import plot_comparison

splits = make_split(y, task, seed=42)
results = run_comparison(
    X, y, make_tabpfn_models(task, seed=42, device="auto"),
    task=task, dataset_name="Your dataset", seed=42, splits=splits,
)
save_results(results, "results/tabpfn-comparison.json")
comparison_table(results)
plot_comparison(results)
```

The constructors return sklearn-compatible adapters for binary/multiclass classification
or regression. The labeled training partition supplies context. Predictions and class
probabilities retain the original target labels and class order, including string labels.
Validation and test labels are used only for metric calculation, never passed to `fit`.

TabPFN's adapter declares native preprocessing, so the default shared workflow passes
its original feature DataFrame through unchanged. This preserves mixed numeric and
categorical inputs and missing values for TabPFN's train-fitted preprocessing. Classical
models keep their shared imputation/one-hot/scaling pipeline. Each override is recorded
on the affected model row; the top-level preprocessing describes the default pipeline.
Explicit `preprocessors={model_name: transformer}` overrides one model, and a global
`preprocessor=` remains an explicit caller choice. If supplying engineered features,
rename the corresponding model row to `TabPFN - Engineered` instead of reporting it as raw.

## Frozen protocol and recorded evidence

- The checkpoint repository, immutable revision, filename, and SHA-256 hash are saved
  with every model result. The identity comes from `checkpoints/manifest.json`.
- The adapter uses the regular inference API, never the fine-tuning API. Context fitting
  and predictions run without gradients, and all loaded network parameters are marked
  nontrainable. `fine_tune=True` is rejected.
- `tuning_config=None`, `eval_metric=None`, and `differentiable_input=False` disable
  optional calibration/threshold tuning and differentiable-input paths. Attempting to
  give the adapter a validation-search grid is rejected before checkpoint loading.
- Runtime metadata includes the resolved device, CUDA device name when applicable,
  resolved ensemble count, Torch thread count, package versions, and active checkpoint
  limits. The complete underlying estimator settings are also saved.
- The adapter preserves library pretraining limits. Missing cached weights, unavailable
  CUDA, CPU row-limit violations, and fitting memory exhaustion raise actionable
  `Unavailable` errors. The demo exits with code 2 and a `Not evaluated` explanation;
  it does not produce substitute scores for an unavailable model.

The pinned checkpoint's default CPU training limit is 5,000 rows. The selected public
datasets' 60% training partitions exceed that limit, so full-data runs need a supported
device such as CUDA. The integration neither drops rows nor turns off limits
automatically. An existing explicit `TABPFN_ALLOW_CPU_LARGE_DATASET` setting is recorded
in runtime metadata; actual memory and runtime feasibility remain dataset-specific.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Real-checkpoint tests compare loaded weights against a separately initialized checkpoint,
then verify unchanged weights after context fitting and prediction, disabled gradients,
no backpropagation or optimizer steps, training-only context, metrics and provenance for
all task types, native mixed inputs, original class mapping, auto ensemble resolution,
and explicit rejection of unsupported configurations. Combined CLI tests run the tree
models and TabPFN on shared splits and regenerate plots/tables from saved artifacts.
Real-inference tests skip with a reason if weights are not cached; they never download
weights as a side effect of running the test suite.
