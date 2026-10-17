# Seoul Bike regression notebook

The [regression notebook](../notebooks/02_regression.ipynb) predicts hourly rented-bike
counts using observed weather and calendar information. Run it from a fresh kernel:

```bash
.venv/bin/python scripts/execute_notebook.py notebooks/02_regression.ipynb --device cuda
```

Install the project dependencies first, following the [runtime setup](model-runtime.md).
Pre-cache the pinned regression checkpoints if needed:

```bash
.venv/bin/python scripts/smoke_models.py --models tabpfn mitra --tasks regression --device cpu --download
```

The notebook downloads the dataset on first execution and defaults to cached weights
(`DOWNLOAD_WEIGHTS=False`). Change that setting explicitly to allow checkpoint downloads.
The execution helper uses the active Python interpreter and saves notebook outputs in
place. Restricted environments may require access to local kernel ports and the GPU.

## Source and prediction goal

[Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
(2020), [DOI 10.24432/C5F62R](https://doi.org/10.24432/C5F62R), is distributed by UCI under
CC BY 4.0. `datasets/seoul-bike.json` pins the download URL, archive/CSV SHA-256 hashes,
CP1252 encoding, schema, target, date format, and source date range.

The loader retains all 8,760 hourly rows in source order, including 295 non-functioning
hours, and excludes `Rented Bike Count` from the 13 predictors. It verifies the complete
hourly sequence from 2017-12-01 through 2018-11-30, numeric finiteness, nonnegative integer
targets, and absence of missing values. Date is parsed with `%d/%m/%Y` and represented
as integer days since the fixed 1970-01-01 epoch for compatibility with every model.
Downloaded archives live in the ignored `.cache/datasets/` directory.

The protocol is **retrospective tabular prediction with observed weather**, using a
seed-42 random 60/20/20 split: 5,256 training, 1,752 validation, and 1,752 test rows.
This follows the project's simple model-comparison goal; it does not evaluate future
forecasting. Neighboring hours and dates can appear in different partitions. Weather
forecasts and their availability at a future horizon are outside this experiment.
A forecasting experiment would require a separate chronological protocol and suitable
predictors; its scores should not be mixed with these results.

All models use the same source rows and split positions. Exploratory analysis uses
training data. Learned preprocessing fits training rows only, and classical model
selection uses validation RMSE. Models are not refitted on train plus validation.

## Features, models, and runtime

Raw foundation-model settings use the source predictors with Date's compatibility
conversion and native train-only preprocessing. Engineered settings keep these and add:

- Hour sine/cosine, with a 24-hour period.
- Weekday (Monday=0), month, and a weekend indicator.
- Day-of-year sine/cosine, with a fixed 365.25-day period.

These row-local functions use no rental counts, target lags, data-fitted origins,
population statistics, or future aggregates. The engineered frame has 20 columns
before encoding. The classical models also use this engineered frame with training-only
imputation, scaling, and categorical encoding.

Seven predeclared rows compare the training-mean dummy, XGBoost, LightGBM, raw TabPFN,
raw Mitra v2, engineered TabPFN, and engineered Mitra v2. Each classical model uses the
shared four-candidate, 80-tree grid, selected by minimum validation RMSE. Foundation
weights remain frozen without tuning: TabPFN uses its auto ensemble; Mitra uses one
estimator and decodes the regression head using the softmax-weighted bin-center mean.
No log target transform, clipping, rounding, or forced-zero postprocessing is applied.

Each recipe runs in a fresh spawned worker that releases model/GPU allocations on exit.
The default budget is 300 seconds per variant, including loading and prediction; change
it with `--model-timeout`. Missing prerequisites, known memory limits, or timeouts create
an explicit **Not evaluated** row with no scores. Unexpected failures retain diagnostics
and fail the notebook's final assertion. There is no automatic subsampling.

The full training partition exceeds TabPFN's default CPU row limit, so CUDA is appropriate
for full execution. Mitra uses bfloat16 autocast on supported CUDA devices, float16 on
other CUDA devices, and float32 on CPU. Checkpoint parameters remain unchanged; device
and precision metadata are saved. GPU precision can produce small numerical differences.

## Outputs and reproduction

Full artifacts are saved under `results/seoul-bike/full/`:

- `results.json`: actual validation/test predictions and metrics, selected parameters,
  validation trials, source metadata, split positions, protocol, versions, and checkpoint
  hashes/runtime details for evaluated foundation models.
- `splits.json`: source row positions, profile, seed, explicit partitions, and protocol.
- `metrics.csv`: RMSE (primary), MAE, R², status, elapsed seconds, and any runtime reason.
- `rmse.png`: the measured primary-metric comparison.

RMSE and MAE use bikes per hour; lower is better. R² is unitless, higher is better, and
may be negative. Empty metrics mean unevaluated, not zero. Plots exclude unavailable
rows while the table retains them. These are descriptive single-split results, without
significance testing or claims of universal model superiority.

Regenerate the table and plot without fitting models:

```python
from src.benchmark import load_results
from src.evaluation import comparison_table
from src.plotting import plot_comparison

results = load_results("results/seoul-bike/full/results.json")
comparison_table(results)
plot_comparison(results).savefig("results/seoul-bike/full/rmse.png", dpi=150)
```

For an explicitly separate 500-row execution check:

```bash
.venv/bin/python scripts/execute_notebook.py notebooks/02_regression.ipynb --profile smoke --device cpu --output /tmp/seoul-bike-smoke.ipynb
```

Smoke artifacts go under `results/seoul-bike/smoke/` and must not be presented as full
benchmark scores. Source validation, day-first parsing, all-row retention, calendar
features, deterministic splits, and baseline replay are covered by offline tests:

```bash
.venv/bin/python -m unittest discover -s tests -p test_seoul_bike.py -v
```

The source-dependent tests require the pinned archive in cache and never download data.
