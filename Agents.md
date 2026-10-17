# AGENTS.md

## Project Overview

This repository benchmarks **zero-shot / tabular foundation models** against **classical machine-learning models** on publicly available tabular datasets.

The primary research question is:

> Can zero-shot tabular models match or outperform well-tuned classical machine-learning models on common tabular learning problems?

The initial zero-shot models include:

- Mitra v2
- TabPFN

The classical model family includes models such as:

- XGBoost
- LightGBM
- scikit-learn baselines where appropriate

Additional models may be added when useful, but do not remove the core benchmark models without an explicit reason.

The repository should remain a **small, reproducible benchmarking project**, not evolve into a web application, API, or unnecessarily complex framework.

---

## Benchmark Tasks

The benchmark uses three publicly available datasets representing three different supervised-learning tasks:

1. Binary classification
2. Regression
3. Multi-class classification

Each dataset should have one primary notebook.

Expected notebook structure:

```text
notebooks/
├── 01_binary_classification.ipynb
├── 02_regression.ipynb
└── 03_multiclass_classification.ipynb
```

Do not combine all three experiments into one large notebook.

---

## Core Experimental Rules

Experimental fairness and reproducibility are more important than achieving the highest possible score.

### Dataset Splitting

Each dataset must be divided into:

- training set
- validation set
- test set

Use a fixed random seed wherever random splitting is involved.

Classification datasets should use stratification whenever appropriate.

The exact same train/validation/test observations must be used when comparing models.

Persist or deterministically recreate the split so results are reproducible.

### Test Set Isolation

The test set is the final holdout set.

Do not:

- tune hyperparameters using test results
- select features based on test performance
- select models based on test performance
- fit preprocessing on the complete dataset
- repeatedly inspect the test set while developing the experiment

Model selection and tuning must use only the training and validation data.

Use the test set only for the final evaluation of the selected configuration.

---

## Prevent Data Leakage

All transformations that learn information from the data must be fitted using training data only.

This includes, where applicable:

- scaling
- normalization
- imputation
- categorical encoding
- target encoding
- feature selection
- dimensionality reduction
- learned feature transformations

Prefer scikit-learn `Pipeline` and `ColumnTransformer` abstractions when they reduce leakage risk.

Never fit preprocessing on the complete dataset before performing the train/validation/test split.

---

## Classical ML Experiments

Classical ML models may use reasonable dataset-specific preprocessing and feature engineering.

Examples include:

- missing-value handling
- categorical encoding
- feature scaling
- transformations
- interaction features
- domain-relevant feature engineering
- feature selection

Feature engineering must be explainable and documented in the notebook.

Classical models may be tuned using the validation set or cross-validation performed only on the training portion.

Do not intentionally weaken classical baselines to make zero-shot models appear better.

Classical models should represent a competent real-world tabular ML workflow.

At minimum, include an appropriate simple baseline so that more sophisticated models can be interpreted in context.

---

## Zero-Shot / Foundation Model Experiments

Where supported, evaluate zero-shot tabular models in two settings.

### Experiment A — Raw Features

Use the dataset with minimal preprocessing required for the model to operate.

Do not add handcrafted predictive features in this experiment.

Basic operations required for compatibility are allowed, for example:

- datatype conversion
- missing-value representation
- column-name normalization
- model-required formatting

### Experiment B — Feature Engineered

Evaluate the zero-shot model using reasonable feature engineering where technically supported.

Use transformations comparable to those available to the classical ML experiment when appropriate.

Keep these results separate from the raw-feature experiment.

Use clear names such as:

```text
TabPFN - Raw
TabPFN - Engineered
Mitra v2 - Raw
Mitra v2 - Engineered
XGBoost
LightGBM
```

Do not silently apply feature engineering to a zero-shot model and report it as the raw zero-shot result.

---

## Hyperparameter Tuning

Hyperparameter tuning is allowed for classical models.

Keep tuning practical enough that another developer can reproduce the experiment on normal hardware.

Prefer:

- sensible documented search spaces
- RandomizedSearchCV
- Optuna
- small GridSearchCV searches

over extremely large brute-force searches.

Record:

- search method
- search space
- optimization metric
- number of trials or configurations
- best parameters

Zero-shot models should use their recommended/default inference configuration unless the experiment explicitly investigates another configuration.

Do not perform extensive tuning of a model and continue describing the result as purely zero-shot.

---

## Evaluation Metrics

Use metrics appropriate for each problem type.

### Binary Classification

Prefer multiple complementary metrics such as:

- ROC-AUC
- F1
- Precision
- Recall
- Accuracy

Where class imbalance exists, include metrics that are robust to imbalance, such as PR-AUC when appropriate.

### Multi-class Classification

Prefer metrics such as:

- Macro F1
- Weighted F1
- Accuracy

Include per-class results or a confusion matrix when useful.

### Regression

Prefer metrics such as:

- RMSE
- MAE
- R²

The primary comparison metric for each dataset must be clearly identified in the notebook.

Do not claim one model is universally superior solely because it wins on one secondary metric.

---

## Final Visual Comparison

Every notebook must end with a clear comparison of the evaluated models.

At minimum, include a plot comparing the primary evaluation metric across models.

For example:

```text
Model                 ROC-AUC
------------------------------
LightGBM                0.91
XGBoost                 0.90
TabPFN - Raw            0.89
TabPFN - Engineered     0.91
Mitra v2 - Raw          0.88
```

Use readable model names and axis labels.

Where several metrics are important, create additional plots or a compact comparison table rather than producing an unreadable chart.

Plots must be generated from evaluation results rather than manually entered values.

---

## Notebook Requirements

The notebooks are the main user-facing artifacts of this repository.

Each notebook should be runnable from top to bottom.

Each notebook should roughly follow this flow:

```text
1. Problem definition
2. Dataset description
3. Imports and configuration
4. Data loading
5. Exploratory data analysis
6. Train/validation/test split
7. Preprocessing
8. Classical ML models
9. Zero-shot models — raw features
10. Zero-shot models — engineered features
11. Final test evaluation
12. Comparison table
13. Comparison plots
14. Observations and conclusions
```

Keep notebooks readable.

Move reusable implementation into Python modules rather than copying large functions between notebooks.

The notebooks should focus on:

- experiment explanation
- dataset exploration
- experiment execution
- visualization
- interpretation

They should not contain hundreds of lines of reusable infrastructure code.

---

## Reusable Python Code

Reusable code should live under `src/`.

Recommended structure:

```text
src/
├── __init__.py
├── data.py
├── preprocessing.py
├── evaluation.py
├── plotting.py
└── models/
    ├── __init__.py
    ├── classical.py
    └── zeroshot.py
```

Create modules only when code is genuinely reused.

Do not introduce abstractions merely for architectural elegance.

Three readable notebooks plus a few reusable modules are preferable to a large framework.

---

## Recommended Repository Structure

Prefer approximately this structure:

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
│   or pyproject.toml
├── notebooks/
│   ├── 01_binary_classification.ipynb
│   ├── 02_regression.ipynb
│   └── 03_multiclass_classification.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── plotting.py
│   └── models/
│       ├── __init__.py
│       ├── classical.py
│       └── zeroshot.py
├── results/
│   └── benchmark_results.md
└── .gitignore
```

Add directories such as `data/`, `configs/`, or `figures/` only when they provide a concrete benefit.

---

## Data Handling

Prefer datasets that can be downloaded programmatically from a stable public source.

Do not commit large raw datasets to Git unless there is a strong reason.

Prefer code that:

1. downloads or retrieves the dataset
2. validates it
3. creates the deterministic train/validation/test split

Document the dataset:

- name
- source
- license if relevant
- number of rows
- number of features
- target variable
- task type

The repository must make it clear how another developer can obtain exactly the same data.

---

## Reproducibility

Experiments should be reproducible wherever the underlying model allows deterministic execution.

Define a shared random seed, for example:

```python
RANDOM_STATE = 42
```

Use it consistently across:

- dataset splitting
- classical models
- hyperparameter optimization
- sampling

Record important package versions.

Do not depend on undocumented local files.

Do not hard-code developer-specific absolute file paths.

The repository should work after a fresh clone when the documented setup procedure is followed.

---

## Result Storage

The aggregate results from all three datasets belong in:

```text
results/benchmark_results.md
```

This document should contain:

- experiment objective
- datasets used
- methodology
- models compared
- feature-engineering strategy
- hyperparameter-tuning strategy
- evaluation metrics
- final results for each dataset
- comparison tables
- important observations
- limitations
- overall conclusion

Include enough information that a reader can understand how the benchmark was performed without reading every notebook.

Do not copy every notebook output into this document.

Focus on the final methodology and conclusions.

---

## Result Integrity

Never fabricate benchmark results.

Never insert plausible-looking metric values when an experiment has not been executed.

If a model cannot run because of:

- hardware constraints
- dependency problems
- model availability
- API restrictions
- dataset limitations

report that explicitly.

Use markers such as:

```text
Not evaluated
```

rather than inventing results.

When updating benchmark results, ensure the numbers correspond to actual notebook outputs.

---

## Result Interpretation

Avoid conclusions such as:

> Model X is better than classical machine learning.

unless the experiments genuinely support that conclusion.

Prefer statements such as:

> TabPFN achieved the highest ROC-AUC on Dataset A while LightGBM performed better on Dataset B.

Discuss:

- predictive performance
- training/inference time where useful
- preprocessing requirements
- feature-engineering requirements
- computational requirements
- implementation complexity
- limitations

The objective is to understand **when zero-shot tabular models are competitive**, not to prove that one model family is superior.

---

## README Requirements

`README.md` is intended for human users and should contain:

- project objective
- models being evaluated
- datasets
- repository structure
- environment setup
- how to run the notebooks
- how to reproduce the benchmark
- link to `results/benchmark_results.md`

Keep detailed agent instructions in `AGENTS.md`, not the README.

---

## Dependencies

Keep dependencies minimal.

Do not introduce:

- FastAPI
- Streamlit
- databases
- Docker
- orchestration frameworks
- cloud infrastructure

unless explicitly required later.

This repository is an experimental benchmarking project.

Prefer standard scientific Python tools such as:

- pandas
- numpy
- scikit-learn
- matplotlib
- xgboost
- lightgbm

plus the libraries required by the zero-shot models.

---

## Coding Style

Use modern Python.

Prefer:

- clear function names
- type hints for reusable functions
- docstrings for non-obvious reusable functions
- small focused functions
- `pathlib.Path` for filesystem operations
- configuration/constants rather than repeated magic values

Avoid:

- unnecessary classes
- premature abstraction
- excessive design patterns
- duplicated experiment logic
- hidden global state

Favor readability over cleverness.

---

## Before Completing Changes

For changes affecting experiments, verify:

1. No train/test leakage was introduced.
2. All models use the same underlying dataset split.
3. Hyperparameter tuning does not use the test set.
4. Random seeds are controlled where possible.
5. Notebook imports still work.
6. Reusable code is not unnecessarily duplicated.
7. Results are generated from actual experiments.
8. Documentation reflects the current experiment.
9. A fresh user can understand how to reproduce the results.

For changes to shared utilities, consider whether all three notebooks are affected.

---

## Agent Behavior

When implementing a task:

- inspect the relevant existing files before changing architecture
- preserve reproducibility and benchmark fairness
- make the smallest reasonable change
- prefer simple solutions
- do not create infrastructure unrelated to the benchmark
- do not silently change datasets, splits, metrics, or benchmark methodology
- explain any methodological change that could affect comparison results
- update documentation when experiment behavior changes

When uncertain between a simpler notebook-oriented implementation and a more elaborate software architecture, prefer the simpler implementation.

The repository exists to produce a **credible, reproducible benchmark**, not to demonstrate application architecture.