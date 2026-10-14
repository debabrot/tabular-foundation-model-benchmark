# Tabular Foundation Model Benchmark

A practical benchmark comparing modern **tabular foundation models** with established machine learning techniques such as XGBoost, LightGBM, CatBoost, Random Forest, and Logistic Regression.

The goal of this project is not just to answer **which model achieves the best predictive performance**, but also to understand the trade-offs between:

- predictive performance
- training and inference time
- hardware requirements
- preprocessing effort
- hyperparameter tuning
- dataset size
- developer experience

The benchmark focuses on three modern tabular foundation models:

- **Mitra-v2**
- **TabFM**
- **TabPFN**

and compares them against widely used classical ML baselines.

---

## Motivation

Gradient-boosted decision trees such as **XGBoost, LightGBM, and CatBoost** have dominated tabular machine learning for years.

Recently, tabular foundation models have introduced a different approach.

Instead of learning every problem entirely from scratch, these models are pretrained across large collections of synthetic or real tabular tasks and attempt to transfer that prior knowledge to unseen datasets.

This raises several interesting questions:

- Can tabular foundation models outperform well-tuned gradient boosting?
- How well do they perform with very little training data?
- How much hyperparameter tuning do traditional models need to remain competitive?
- What is the runtime and hardware cost of foundation models?
- Are the gains large enough to justify their additional computational requirements?
- Which approach would be more practical for a real-world ML project?

This repository explores these questions experimentally.

---

# Models

## Tabular Foundation Models

### Mitra-v2

Mitra-v2 is a compact tabular foundation model designed for classification and regression tasks.

One interesting aspect of Mitra-v2 is its relatively small model size compared with many other foundation models, making it an interesting candidate for evaluating the trade-off between model size and predictive performance.

### TabPFN

TabPFN uses a pretrained transformer architecture designed specifically for tabular prediction.

Instead of performing traditional iterative training on every dataset, it uses knowledge acquired during pretraining to perform inference on new tabular tasks.

### TabFM

TabFM is a large tabular foundation model designed to perform prediction using pretrained representations and in-context information from the target dataset.

Its significantly larger scale also makes it useful for studying whether additional model capacity translates into meaningful gains on practical tabular problems.

---

## Classical ML Baselines

The benchmark includes the following commonly used algorithms:

- Logistic Regression / Ridge Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

For gradient-boosting models, both **default configurations** and **hyperparameter-tuned configurations** may be evaluated.

This allows comparison between:

> Foundation models with minimal task-specific tuning

and

> Traditional ML models after explicit optimization for the target dataset.

---

# Experimental Questions

The experiments are designed around several research questions.

## 1. Predictive Performance

How do foundation models compare against established tree-based algorithms?

Depending on the task, metrics include:

### Classification

- ROC-AUC
- Accuracy
- F1 Score
- Log Loss
- Macro F1 for multiclass problems

### Regression

- RMSE
- MAE
- R²

---

## 2. Performance in Low-Data Settings

Foundation models should theoretically benefit from knowledge acquired during pretraining.

To test this, selected datasets are evaluated using different amounts of training data.

Example:

```text
100 samples
500 samples
1,000 samples
5,000 samples
10,000 samples
Full dataset
```

This experiment investigates whether foundation models provide a meaningful advantage when labeled data is scarce.

---

## 3. Hyperparameter Tuning vs Foundation Models

Gradient-boosting models often improve significantly after tuning.

The benchmark therefore compares configurations such as:

```text
XGBoost — default
XGBoost — 10 tuning trials
XGBoost — 50 tuning trials
XGBoost — 100 tuning trials
```

against foundation models with little or no task-specific tuning.

This helps answer:

> How much tuning effort is required before classical ML matches or exceeds a pretrained tabular foundation model?

---

## 4. Runtime

Predictive performance alone does not determine whether a model is practical.

The benchmark records:

- preprocessing time
- training time
- inference time
- total experiment time

Foundation models and traditional ML models have fundamentally different compute profiles, so this comparison is particularly important.

---

## 5. Hardware Requirements

Where possible, experiments also record:

- CPU utilization
- system RAM
- GPU requirement
- peak GPU VRAM
- model/checkpoint size

This provides a more realistic picture of the infrastructure required to use each model.

---

## 6. Developer Experience

Some differences cannot be represented by a single metric.

The project also compares:

- preprocessing requirements
- categorical feature handling
- missing-value handling
- feature scaling requirements
- hyperparameter tuning effort
- installation complexity
- GPU requirements
- API usability

The objective is to evaluate the **overall effort required to obtain a strong tabular model**, rather than focusing exclusively on leaderboard performance.

---

# Datasets

The benchmark uses multiple datasets covering different problem characteristics.

The final dataset collection will include examples of:

| Dataset Type | Purpose |
|---|---|
| Small binary classification | Evaluate low-data performance |
| Medium binary classification | General benchmark |
| Multiclass classification | Evaluate multiclass capability |
| Imbalanced classification | Evaluate minority-class performance |
| Small regression | Foundation-model performance with limited data |
| Medium regression | Scaling behaviour |
| Mixed categorical/numerical | Evaluate preprocessing and categorical handling |

Potential datasets include:

- Adult Income
- Breast Cancer Wisconsin
- Bank Marketing
- Dry Bean
- California Housing
- Concrete Compressive Strength
- additional OpenML datasets

The exact dataset list may evolve as the benchmark develops.

---

# Experimental Design

To ensure a fair comparison, all models use the same dataset partitions.

Where computationally feasible, experiments use:

```text
Repeated train/test splits
or

K-fold cross-validation
```

with fixed random seeds.

Results are reported as averages along with variability where appropriate.

Example:

```text
ROC-AUC: 0.912 ± 0.004
```

rather than relying on a single potentially lucky train/test split.

---

# Repository Structure

```text
tabular-foundation-model-benchmark/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_classical_ml.ipynb
│   ├── 03_tabpfn.ipynb
│   ├── 04_mitra_v2.ipynb
│   ├── 05_tabfm.ipynb
│   └── 06_results_analysis.ipynb
│
├── src/
│   ├── data.py
│   ├── metrics.py
│   ├── benchmark.py
│   └── utils.py
│
├── results/
│   ├── benchmark_results.csv
│   └── runtime_results.csv
│
└── figures/
    ├── performance_comparison.png
    ├── runtime_comparison.png
    └── low_data_comparison.png
```

---

# Benchmark Output

Each experiment produces a standardized result record.

Example:

```text
dataset: adult
task: classification
model: xgboost
seed: 42

roc_auc: 0.912
f1: 0.847
accuracy: 0.869
log_loss: 0.314

training_seconds: 4.32
prediction_seconds: 0.04

peak_ram_mb: 1420
peak_vram_mb: 0

hyperparameter_tuning: true
```

Results from all experiments are combined into a common dataset for downstream analysis and visualization.

---

# Planned Comparisons

The benchmark will eventually include comparisons such as:

### Overall model performance

| Model | ROC-AUC | F1 | Training Time | Inference Time |
|---|---:|---:|---:|---:|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD |
| CatBoost | TBD | TBD | TBD | TBD |
| TabPFN | TBD | TBD | TBD | TBD |
| Mitra-v2 | TBD | TBD | TBD | TBD |
| TabFM | TBD | TBD | TBD | TBD |

---

# Key Trade-offs

Beyond accuracy, the final analysis will evaluate models across several dimensions.

| Model | Tuning | Preprocessing | GPU | Training Required | Compute Cost |
|---|---|---|---|---|---|
| Logistic Regression | Low | High | No | Yes | Low |
| Random Forest | Low | Medium | No | Yes | Low |
| XGBoost | Medium/High | Medium | Optional | Yes | Medium |
| LightGBM | Medium/High | Medium | Optional | Yes | Medium |
| CatBoost | Medium | Low | Optional | Yes | Medium |
| TabPFN | Low | Low | Recommended | Minimal | TBD |
| Mitra-v2 | Low | Low | Recommended | Minimal | TBD |
| TabFM | Low | Low | Recommended | Minimal | TBD |

Actual conclusions will be updated after completing the benchmark.

---

# Reproducing the Experiments

Clone the repository:

```bash
git clone https://github.com/<username>/tabular-foundation-model-benchmark.git

cd tabular-foundation-model-benchmark
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter lab
```

---

# Hardware

Experiments will document the hardware used for each benchmark run.

This is particularly important for foundation models because GPU memory requirements and inference cost may differ significantly from traditional ML algorithms.

Hardware details will be added alongside the final results.

---

# Results

🚧 **Work in progress**

The benchmark is currently being developed.

Results, visualizations, and detailed analysis will be added as experiments are completed.

Planned visualizations include:

- average model ranking across datasets
- performance vs dataset size
- performance vs training-data size
- accuracy vs runtime
- accuracy vs tuning budget
- GPU memory consumption
- inference latency
- model performance across dataset types

---

# Blog

A detailed article based on the experiments will accompany this repository.

The article will focus on the broader question:

> **Are tabular foundation models ready to replace XGBoost, LightGBM, and CatBoost for practical machine learning?**

Rather than identifying a single universal winner, the goal is to understand **when each approach makes sense**.

---

# Disclaimer

This project is intended as an independent practical benchmark.

Foundation models differ significantly in architecture, pretraining strategy, supported dataset sizes, licensing, and computational requirements.

Results should therefore be interpreted within the constraints of the datasets, hardware, model versions, and experimental settings used in this repository.

---

# License

The benchmark code in this repository is released under the MIT License unless otherwise specified.

Individual models and datasets may have their own licenses and usage restrictions. Please refer to their respective repositories and documentation before commercial use.
