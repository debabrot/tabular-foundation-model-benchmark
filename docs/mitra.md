# Frozen Mitra v2 comparisons

Run the offline synthetic comparison for binary classification, multiclass classification,
and regression:

```bash
.venv/bin/python scripts/demo_baseline.py --mitra --device cpu
```

Results, predictions, checkpoint metadata, CSV tables, and PNG plots are saved under
`results/mitra-demo/`. These are synthetic integration examples, not scores for Online
Shoppers, Seoul Bike, or Dry Bean. Add `--download` if the pinned weights are not cached.
Both task-specific checkpoints are recorded in `checkpoints/manifest.json`.

Combine all integrated model families on the same split:

```bash
.venv/bin/python scripts/demo_baseline.py --classical --tabpfn --mitra --device cpu
```

Combined artifacts go to `results/combined-demo/`. Add `--render-only` to either command
to recreate tables and plots from saved results without loading or fitting models.
Use `--mitra-batch-size` to change query batching or `--output-dir` for another directory.

## Use in notebooks

```python
from src.benchmark import run_comparison, save_results
from src.data import make_split
from src.models.mitra import make_mitra_models
from src.evaluation import comparison_table
from src.plotting import plot_comparison

splits = make_split(y, task, seed=42)
results = run_comparison(
    X, y, make_mitra_models(task, seed=42, device="auto"),
    task=task, dataset_name="Your dataset", seed=42, splits=splits,
)
save_results(results, "results/mitra-comparison.json")
comparison_table(results)
plot_comparison(results)
```

Mitra declares native preprocessing, so the shared workflow passes it original feature
columns by default. The adapter learns ordinal category encodings on training rows;
unknown and missing categories become NaN. AutoGluon's native Mitra preprocessor then
imputes training means, removes constant columns, and applies seeded feature mirrors.
The network's quantile embedding derives quantiles and scaling from training support.
Dates must first be converted into explicit features. Global or per-model preprocessing
overrides remain available; name engineered experiments `Mitra v2 - Engineered`.

Classification labels are encoded from training data and decoded back to their original
values. Probability columns follow the saved class order. The classifier uses the first
K logits of the 10-class checkpoint head and applies a stable softmax.

## Distributional regression without fine-tuning

The stock installed AutoGluon sklearn regressor is configured for a scalar head.
The v2 checkpoint instead has 1,000 bins. The adapter uses AutoGluon's existing `Tab2D`
architecture directly and validates the checkpoint's task and head configuration.
It never constructs `TrainerFinetune`, an optimizer, a bagged fit, or a validation split.

For regression, training targets are min-max normalized, optionally mirrored with the
seeded native transformation, and bucketed against 1,001 edges spanning -0.5 to 1.5.
The forward pass produces 1,000 logits per query row. Their softmax-weighted bin-center
mean is mapped back through the mirror and training target range to original units.
The result is a point prediction suitable for RMSE, MAE, and R²; it is not the argmax bin.

The binning and mean-decoding convention is verified against the
[official regression implementation at revision b4701e8](https://huggingface.co/autogluon/mitra-finetune/blob/b4701e8148dc33b00ed15d7086ff59816957cde4/src/mitra_finetune/patches.py)
and the [Mitra v2 regressor model card](https://huggingface.co/autogluon/mitra-regressor-2).
Only the installed architecture and native preprocessing are used at runtime; no
downloaded Python code or process-wide patches are executed. The `mitra-finetune`
package and its CUDA fine-tuning recipe are not dependencies of this frozen adapter.

## Fixed inference choices and limits

- One frozen checkpoint per task, one estimator, and
  seed 42 by default. There is no fine-tuning, class/feature shuffle, feature selection,
  threshold calibration, or validation search. Explicit search grids and
  `fine_tune=True` are rejected.
- CPU inference uses float32. CUDA uses bfloat16 autocast when supported, matching the
  native AutoGluon GPU inference setting; older CUDA devices fall back to float16
  autocast. Checkpoint parameters remain unchanged in float32, while logits are
  converted back to float32 before probability/mean decoding. The resolved precision
  is recorded, and device-dependent numerical differences are expected.
- Feature and target mirrors follow the native defaults but draw from a local seeded
  RNG. Temporary network initialization preserves the caller's Torch RNG state.
  Parameters are nontrainable, the network is in evaluation mode, and forward passes
  run without gradients.
- Every training row supplies context in the saved order. The default 8,192-row support
  guard rejects larger inputs instead of silently sampling or reducing context after
  memory errors. Query rows are processed in batches of 256 by default. Changing query
  batch size changes memory use, not the chosen training support.
- The default feature guard is 500 input columns; the classifier supports 2–10 classes.
  Regression requires finite, varying training targets. Unsupported schemas, mismatched
  prediction columns, missing weights, unavailable CUDA, and memory errors produce
  explicit errors. No replacement model or fabricated score is reported.
- CPU is supported for the tiny fixtures. Full-dataset memory/runtime feasibility is
  not established by these examples; CUDA is recommended for larger contexts. The
  integration uses AutoGluon internals, with the installed version saved for provenance.

Saved runtime metadata includes both checkpoint-file hashes, immutable revision,
architecture/head configuration, package versions, resolved device, precision, training
support size, query batch size, mean imputation, dropped-column mask, mirror settings,
target range, and regression decoding rule. Saved shared split indices, dataset
fingerprints, predictions, probabilities, and metrics connect each score to its inputs.
Fitted context and network weights are not copied into the JSON artifacts.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Real-checkpoint tests verify unchanged classifier and regressor weights, no optimizer,
trainer, or backward call, preserved caller data/RNG state, train-only context, class
alignment, mean decoding and target-unit recovery, reproducible query batching,
round-trip serialization, and repeated-process demo outputs. Unsupported settings are
tested before checkpoint loading. Cached-weight tests skip with a reason when weights
are absent; running tests does not initiate downloads.

The runtime smoke runner now uses this adapter for Mitra, including regression:

```bash
.venv/bin/python scripts/smoke_models.py --models mitra --device cpu
```
