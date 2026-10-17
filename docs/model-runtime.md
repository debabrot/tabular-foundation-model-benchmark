# Model runtime verification

The runtime probe checks predictions, not public-dataset evaluation or benchmark scores.
Run it with Python 3.12 from the repository root:

```bash
.venv/bin/python scripts/smoke_models.py
```

The default is offline. To download missing, pinned checkpoints:

```bash
.venv/bin/python scripts/smoke_models.py --download --timeout 300
```

Use `--models xgboost lightgbm`, `--tasks binary`, or `--device cpu` to narrow a run.
The device option applies to foundation models; classical smoke tests always use CPU.
The timeout is per model/task, including imports and downloads. Increase it if downloads
or CPU inference need longer. Each check runs in its own process so a failure or timeout
does not prevent the remaining checks from being reported.

The report is written to `results/runtime-smoke.json` (override with `--output`).
It records package versions, Python, CUDA visibility, device, seed, inference settings,
checkpoint revisions and file hashes when loaded, prediction shapes, and any failures.
Exit codes: **0** means all requested checks passed; **1** means a check failed;
**2** means at least one prerequisite was unavailable and no check failed.
An unavailable model is never counted as verified.

## Recorded verification

On 2026-10-15, after ticket 5, the full CPU run completed with **12 passed checks**:
XGBoost, LightGBM, TabPFN-3.5-Fast, and Mitra v2 each passed binary classification,
multiclass classification, and regression. The full suite passed all 28 tests.
An earlier probe identified the incompatible stock Mitra regression path; the current
runner uses the verified frozen v2 adapter described below instead.

The full dataset notebooks subsequently completed all **21 model variants** on the
recorded CUDA GPU. See the [final measured report](../results/benchmark_results.md)
for dataset-specific scores, wall times, hardware, and limitations. Ticket 9 verification
on 2026-10-16 passed all 42 existing tests and three report-integrity tests, with no skips;
`scripts/build_report.py --check` also verified the full saved artifacts and current report.

The sandbox hides CUDA while an earlier unrestricted probe could see it. Device availability in
the JSON report describes the process that ran the checks; it is not inferred from
whether a CUDA-enabled PyTorch build is installed. See `results/runtime-smoke.json`
for the complete resolved versions, checkpoint hashes, and per-check evidence.

## Reproduce the environment

For a fresh environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

The existing `requirements-lock.txt` records the resolved local environment, including
`torch==2.13.0+cu132`. Replaying that file requires a matching platform and PyTorch wheel
index; it is not a portable CPU lockfile. For CPU-only installations or a different CUDA
runtime, install the appropriate build using the [official PyTorch installer](https://pytorch.org/get-started/locally/),
then install the requirements and retain the smoke report's resolved versions.
Reproducing versions does not guarantee identical floating-point outputs across devices.

## Model choices and access

| Model | Smoke-test selection | Hardware and access |
| --- | --- | --- |
| XGBoost | Five trees, depth two, seed 42, one CPU thread | CPU; no pretrained checkpoint |
| LightGBM | Five trees, depth two, seed 42, deterministic mode, one CPU thread | CPU; no pretrained checkpoint |
| TabPFN | **TabPFN-3.5-Fast**, one estimator, seed 42, explicit local checkpoint | CPU supported for these tiny fixtures; GPU recommended for the full datasets; requires model access |
| Mitra v2 classification | Explicit v2 classifier checkpoint, one estimator, seed 42, forward-only frozen inference | CPU for tiny fixtures; CUDA recommended at benchmark sizes |
| Mitra v2 regression | Explicit v2 regressor checkpoint, 1,000-bin head with softmax-weighted mean decoding | Verified frozen CPU inference for tiny fixtures; CUDA recommended at benchmark sizes |

Checkpoint repositories, immutable commit revisions, and required filenames are in
`checkpoints/manifest.json`. Downloads use the ignored repository-local `.cache/`
directory. Files are hashed in each successful checkpoint load. TabPFN is selected
explicitly rather than inheriting the installed library's changing default version.
The runner materializes files with their original names: the installed TabPFN loader
resolves Hub symlinks to extensionless blobs and would otherwise choose the wrong format.

For TabPFN, review the [model card and access terms](https://huggingface.co/Prior-Labs/tabpfn_3_5).
If the Hub requests access, accept the terms yourself and provide an authorized `HF_TOKEN`
environment variable. The runner does not accept terms or open an interactive login.
Public Mitra v2 weights are [Apache-2.0](https://huggingface.co/autogluon/mitra-classifier-2).

The [Mitra v2 regressor card](https://huggingface.co/autogluon/mitra-regressor-2)
specifies a 1,000-bin distributional head, which stock AutoGluon's scalar regressor
does not wire up. Ticket 5 resolves this through a
[forward-only adapter](mitra.md) using the existing architecture and native preprocessing,
with the official binning and mean-decoding convention. It neither substitutes Mitra v1
nor invokes the separate fine-tuning recipe. The smoke runner shares this adapter with
the comparison workflow.

## What is checked

Every model/task receives the same deterministic synthetic fixture: 90 rows, four numeric
features, and a 54/18/18 train/validation/test split, stratified for classification.
All models fit training data only; no validation or test labels are passed during fitting.
Classification checks require 18 valid labels and finite, normalized probabilities in
the expected two- or three-class order. Regression requires 18 finite point predictions.
Small tree/ensemble counts keep this a runtime probe; they are not selected benchmark settings.

Run the verification tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

These check deterministic, disjoint fixtures, malformed prediction rejection, actual
classical inference for all task types, repeated-process predictions, and the unavailable-model
exit/report contract. A successful tiny fixture does not establish memory or runtime
feasibility for the full datasets; the three executed notebooks provide that separate
integration evidence on the recorded GPU.
