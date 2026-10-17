# Tabular foundation model benchmark

Can a pretrained tabular model compete with a classical machine-learning model on
an unfamiliar dataset? This project explores that question across three common tasks:
binary classification, regression, and multiclass classification.

We compare **Mitra v2** and **TabPFN-3.5-Fast** with **XGBoost** and **LightGBM**.
The aim is to understand how their predictive performance compares, how much feature
engineering helps, and what it takes to run them. Each experiment has its own notebook
with an explanation of the task, measured results, and comparison plots.

All **21 model variants** have been evaluated on the full datasets. You can read the
[results report](results/benchmark_results.md) or browse the executed notebooks without
running the experiments yourself.

## Methodology

### Datasets and tasks

The benchmark uses three public UCI datasets. Feature counts below refer to the
original inputs, before encoding or feature engineering.

| Dataset | Prediction task | Rows | Features | Primary metric | Notebook |
| --- | --- | ---: | ---: | --- | --- |
| Online Shoppers Purchasing Intention | Will a browsing session end in a purchase? | 12,330 | 17 | ROC-AUC ↑ | [Binary classification](notebooks/01_binary_classification.ipynb) |
| Seoul Bike Sharing Demand | How many bikes are rented in an hour? | 8,760 | 13 | RMSE ↓ | [Regression](notebooks/02_regression.ipynb) |
| Dry Bean | Which of seven bean varieties does a sample belong to? | 13,611 | 16 | Macro-F1 ↑ | [Multiclass classification](notebooks/03_multiclass_classification.ipynb) |

Higher ROC-AUC and macro-F1 scores are better; lower RMSE is better. The notebooks
also report supporting metrics, such as precision and recall for binary classification,
MAE and R² for regression, and accuracy and weighted-F1 for multiclass classification.

### A shared evaluation protocol

Every model receives the same **60% training / 20% validation / 20% test split**, using
random seed 42. Classification splits preserve class proportions. All source rows are
retained, and any preprocessing that learns from data is fitted on training rows only.

Training data is used to fit classical models and provide context for foundation models.
Validation data selects the classical models' hyperparameters. The test set is held back
for final evaluation and does not influence preprocessing or model selection.

The Seoul Bike experiment uses a random split and observed weather to study
retrospective prediction. Its results do not measure future-demand forecasting, which
would require a chronological split and inputs available at forecast time.

### Models and feature engineering

**XGBoost and LightGBM** use documented, dataset-specific features and a small search
of four hyperparameter configurations each. The best configuration is selected using
the validation set. A simple dummy model provides a reference score for each task.

**Mitra v2 and TabPFN** use frozen pretrained weights, with labeled training examples
supplied as context. Here, “zero-shot” means no task-specific fine-tuning of those
weights. Each foundation model is evaluated in two settings:

- **Raw:** original features with the formatting needed for the model to run.
- **Engineered:** original features plus the same kinds of handcrafted features used
  by the classical models, such as browsing-session ratios or calendar features.

This gives seven evaluated variants per dataset: a dummy baseline, two classical
models, and four foundation-model settings. Dataset splits, checkpoint identities,
selected parameters, predictions, package versions, and timings are saved with the
results so that each score can be traced back to its experiment.

These comparisons describe performance on one fixed split per dataset. They do not
include significance testing or establish that one model is best across all tabular tasks.

## How to run

The commands below assume a Linux or WSL shell and should be run from the repository
root. Use **Python 3.12** and a **CUDA-capable GPU** to reproduce all full-data results.
The reference runs used an NVIDIA RTX 3050 Laptop GPU with 6 GB of memory; classical
models run on CPU. The full training sets exceed TabPFN's default CPU limit.

### 1. Set up the environment

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

For platform-specific PyTorch installation, see the
[environment and model-access guide](docs/model-runtime.md).
[requirements-lock.txt](requirements-lock.txt) records the exact reference environment,
including its CUDA-specific PyTorch build; it requires a compatible platform and wheel
index. Allow roughly 5.8 GB for the environment and 900 MB for model weights, plus space
for datasets and download caches. Small numerical differences across hardware are possible.

### 2. Download and check the model weights

```bash
.venv/bin/python scripts/smoke_models.py --models tabpfn mitra --device cpu --download --timeout 300
```

This downloads missing weights at the revisions pinned in
[the checkpoint manifest](checkpoints/manifest.json), then checks inference on small
synthetic examples. Weights and dataset downloads are stored locally in `.cache/`
and excluded from Git.

If TabPFN requires access approval, accept its terms and supply an authorized
`HF_TOKEN` environment variable as explained in the
[access guide](docs/model-runtime.md). The check writes `results/runtime-smoke.json`;
exit code 0 indicates success, 1 a failure, and 2 a missing prerequisite or runtime limit.

### 3. Run the three experiments

Run these commands one at a time. Each starts a fresh notebook kernel and saves the
executed notebook, including its outputs, in place.

```bash
.venv/bin/python scripts/execute_notebook.py notebooks/01_binary_classification.ipynb --device cuda
.venv/bin/python scripts/execute_notebook.py notebooks/02_regression.ipynb --device cuda
.venv/bin/python scripts/execute_notebook.py notebooks/03_multiclass_classification.ipynb --device cuda
```

Datasets download automatically on the first run. The notebooks expect model weights
to have been cached in step 2. Each model variant has a 300-second budget for loading,
tuning, and prediction; use `--model-timeout` to change it. In the reference runs, model
wall times summed to approximately 7 minutes for Online Shoppers, 2 minutes for Seoul
Bike, and 8 minutes for Dry Bean, excluding downloads and notebook overhead.

To explore interactively, launch JupyterLab and select the project's Python environment:

```bash
.venv/bin/python -m jupyter lab
```

For a smaller CPU execution check, use the 500-row smoke profile:

```bash
.venv/bin/python scripts/execute_notebook.py notebooks/03_multiclass_classification.ipynb --profile smoke --device cpu --output /tmp/dry-bean-smoke.ipynb
```

Smoke results are saved separately from the full benchmark. Mitra may exceed its CPU
budget even in this profile. A model that cannot run is marked **Not evaluated** with
a reason; unexpected errors are marked **Failed** and fail the notebook's final check.

### 4. Rebuild and verify the results report

```bash
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/build_report.py --check
.venv/bin/python -m unittest discover -s tests -v
```

The report builder uses saved predictions to verify metrics and rebuild
[the results report](results/benchmark_results.md), without downloading data or fitting
models again. `--check` confirms that the report is current without changing it.
Tests that require source data or real model weights skip when those caches are absent.

## Where to find the results

The [results report](results/benchmark_results.md) brings together the methodology,
score tables, plots, runtime comparisons, and limitations. For dataset-specific details:

| Experiment | Reproduction guide | Saved results |
| --- | --- | --- |
| Online Shoppers | [Guide](docs/online-shoppers.md) | [Artifacts](results/online-shoppers/README.md) |
| Seoul Bike | [Guide](docs/seoul-bike.md) | [Artifacts](results/seoul-bike/README.md) |
| Dry Bean | [Guide](docs/dry-bean.md) | [Artifacts](results/dry-bean/README.md) |

Each `results/<dataset>/full/` directory contains metric CSVs, comparison plots, saved
splits, and JSON files with predictions and experiment metadata. Dry Bean also includes
per-class scores and confusion matrices. Keep these files together: they support report
verification and allow plots to be regenerated without rerunning the models.

## Repository layout

```text
notebooks/             One executed notebook per prediction task
src/                   Shared data, preprocessing, evaluation, and model code
scripts/               Notebook runner, model checks, demos, and report builder
tests/                 Reproducibility, leakage, model, and result checks
docs/                  Dataset guides and implementation details
datasets/              Dataset source manifests and checksums
checkpoints/           Pinned model manifest; weights are downloaded separately
results/               Measured results, plots, predictions, and summary report
requirements*.txt      Dependency ranges and reference environment versions
```

For more on the implementation, see the [shared workflow](docs/baseline-workflow.md),
[classical models](docs/classical-models.md), [TabPFN](docs/tabpfn.md), and
[Mitra](docs/mitra.md). Small synthetic examples can be generated with
`.venv/bin/python scripts/demo_baseline.py`; their outputs are excluded from Git.
