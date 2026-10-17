# Dry Bean multiclass notebook

Run the [multiclass notebook](../notebooks/03_multiclass_classification.ipynb) from a fresh kernel:

```bash
.venv/bin/python scripts/execute_notebook.py notebooks/03_multiclass_classification.ipynb --device cuda
```

Install the project dependencies using the [runtime setup guide](model-runtime.md).
The data loader uses SciPy's ARFF reader, so no spreadsheet dependency is required.
If classification checkpoints are not already cached, fetch and check the pinned weights:

```bash
.venv/bin/python scripts/smoke_models.py --models tabpfn mitra --tasks multiclass --device cpu --download
```

The notebook defaults to `DOWNLOAD_DATA=True` and `DOWNLOAD_WEIGHTS=False`. Change the
latter explicitly to permit weight downloads, or use the command above. The execution
helper uses the active interpreter and writes notebook outputs in place. Restricted
environments may require permission for local Jupyter kernel ports and GPU access.

## Source and protocol

[Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), Koklu and Ozkan (2020),
[DOI 10.24432/C50S4B](https://doi.org/10.24432/C50S4B), is distributed by UCI under
CC BY 4.0. The manifest `datasets/dry-bean.json` pins the archive URL, archive and ARFF
SHA-256 hashes, exact schema, target, row count, and per-class counts. The selected file
is `DryBeanDataset/Dry_Bean_Dataset.arff`; archives stay in ignored `.cache/datasets/`.

The loader retains all 13,611 rows and 16 numerical predictors in source order, excludes
`Class` from features, and decodes class labels to their original strings. Source names,
including `AspectRation` and `roundness`, remain unchanged. No spreadsheet row indices
or metadata columns are used. The loader validates exact column order, positive finite
predictors, absence of missing values, and these seven target counts:

| Class | Rows |
| --- | ---: |
| BARBUNYA | 1,322 |
| BOMBAY | 522 |
| CALI | 1,630 |
| DERMASON | 3,546 |
| HOROZ | 1,928 |
| SEKER | 2,027 |
| SIRA | 2,636 |

The seed-42 stratified 60/20/20 split has 8,166 training, 2,722 validation, and 2,723 test
observations, with all seven classes in every partition. Every model uses identical
source observations and split positions. Exploration and learned preprocessing use
training data only; classical selection uses validation macro-F1. Models are not refit
on training plus validation, and test scores never change the predeclared recipes.

All 68 identical full records are retained. Disjoint row positions do not guarantee
independent measurements when duplicates span partitions; source identifiers are not
available for group splitting. This limitation is recorded in the notebook rather than
silently changing the dataset or deduplicating it.

## Models and features

Seven rows compare the dummy training-prior classifier, XGBoost, LightGBM, raw TabPFN,
raw Mitra v2, engineered TabPFN, and engineered Mitra v2. Raw settings use the 16 original
measurements. Engineered settings keep these and add natural logs of area, perimeter,
major-axis length, and minor-axis length: four fixed, row-local size representations.
They compress multiplicative scale variation but add no new measured information.
No target aggregates, learned feature selection, or test-selected transformations are used.

The classical models use the 20-column engineered frame with train-only imputation,
scaling, and the shared four-candidate, 80-tree validation grid. Both grids maximize
macro-F1. Foundation models use native preprocessing, pinned frozen weights, and no
validation tuning or fine-tuning. TabPFN uses the checkpoint's auto ensemble; Mitra uses
one estimator. No resampling, class weighting, or threshold optimization is performed.

Each model runs in a separate spawned process, releasing GPU allocations on exit.
The default 300-second budget includes loading, tuning, and prediction; change it with
`--model-timeout`. Missing prerequisites, known memory limits, or timeouts produce
**Not evaluated** with a reason and no metric values. Unexpected failures are distinct
and fail the final notebook assertion after diagnostic artifacts are saved. No context
rows are silently dropped or replaced with a smaller model's predictions.

The full training context fits Mitra's configured 8,192-row limit and exceeds TabPFN's
default CPU row limit, so CUDA is appropriate for full execution. Mitra uses bfloat16
autocast on supported CUDA devices, float16 on other CUDA devices, and float32 on CPU.
Checkpoint parameters remain frozen; runtime metadata records the actual device and
precision. Small numerical differences across devices are possible.

## Outputs

The notebook displays macro-F1 (primary), weighted-F1, accuracy, per-class F1, and a
macro-F1 comparison plot. Full artifacts live in `results/dry-bean/full/`:

- `results.json`: predictions, probability-column class order, metrics, selected settings,
  validation trials, preprocessing recipes, protocol, source rows/splits, package versions,
  checkpoint hashes, device details, and runtime limitations.
- `splits.json`: exact positional partitions, source rows, profile, protocol, and seed.
- `metrics.csv`: all model rows, overall metrics, status, elapsed seconds, and any reason.
- `per_class_metrics.csv`: precision, recall, F1, and support for every evaluated model/class.
- `confusion_matrices.json`: original labels and counts with rows=true and columns=predicted.
- `macro_f1.png`: the primary comparison, generated from measured metrics.

Macro-F1 weights each class equally; weighted-F1 weights by test support. Higher values
are better for all three overall metrics. Empty metrics mean unevaluated, not zero;
unavailable rows are excluded from plots but remain in the status table.

Regenerate the table and plot from saved results without model fitting:

```python
from src.benchmark import load_results
from src.evaluation import comparison_table
from src.plotting import plot_comparison

results = load_results("results/dry-bean/full/results.json")
comparison_table(results)
plot_comparison(results).savefig("results/dry-bean/full/macro_f1.png", dpi=150)
```

An optional fixed, stratified 500-row execution check uses separate artifacts:

```bash
.venv/bin/python scripts/execute_notebook.py notebooks/03_multiclass_classification.ipynb --profile smoke --device cpu --output /tmp/dry-bean-smoke.ipynb
```

Those outputs go under `results/dry-bean/smoke/` and must not be mixed with full results.
Offline tests exercise both hashes, first-use cache handling, exact class labels/counts,
deterministic stratification, log features, original-label prediction/probability alignment,
and saved-result replay:

```bash
.venv/bin/python -m unittest discover -s tests -p test_dry_bean.py -v
```

Source-dependent tests require the pinned archive in cache and never access the network.
Conclusions are descriptive for one split of released image-derived measurements; they
do not establish significance, collection-site transfer, image-segmentation quality, or
universal superiority of a model family.
