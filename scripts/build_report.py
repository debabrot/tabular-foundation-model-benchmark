"""Rebuild the final Markdown report from verified full-data benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import load_results
from src.evaluation import PRIMARY_METRICS, comparison_table, evaluate_predictions

DATASETS = (
    ("online-shoppers", "Online Shoppers", "binary", "01_binary_classification.ipynb", "roc_auc.png"),
    ("seoul-bike", "Seoul Bike", "regression", "02_regression.ipynb", "rmse.png"),
    ("dry-bean", "Dry Bean", "multiclass", "03_multiclass_classification.ipynb", "macro_f1.png"),
)
LABELS = {"roc_auc": "ROC-AUC", "average_precision": "Average precision", "f1": "F1",
          "precision": "Precision", "recall": "Recall", "accuracy": "Accuracy",
          "rmse": "RMSE", "mae": "MAE", "r2": "R²", "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1"}


def verify_result(result: dict, manifest: dict, task: str) -> None:
    """Reject partial profiles and scores inconsistent with the saved predictions."""
    dataset = result["dataset"]
    if dataset.get("profile") != "full" or dataset.get("kind") != "dataset":
        raise ValueError("The final report accepts full public-dataset results only")
    if result["task"] != task or result["primary_metric"] != PRIMARY_METRICS[task] or result["seed"] != 42:
        raise ValueError("Task, primary metric, or seed differs from the benchmark protocol")
    if any(dataset["source"].get(key) != value for key, value in manifest.items()):
        raise ValueError("Recorded dataset source differs from its pinned manifest")
    rows = manifest["rows"]
    if dataset["source_row_positions"] != list(range(rows)):
        raise ValueError("Full results must retain every source row in order")
    positions = np.concatenate([result["splits"][key] for key in ("train", "validation", "test")])
    if positions.dtype.kind not in "iu" or not np.array_equal(np.sort(positions), np.arange(rows)):
        raise ValueError("Result splits must be disjoint and cover the complete source")
    dummy = "Dummy mean" if task == "regression" else "Dummy prior"
    expected = [dummy, "XGBoost", "LightGBM", "TabPFN - Raw", "Mitra v2 - Raw",
                "TabPFN - Engineered", "Mitra v2 - Engineered"]
    if [row["name"] for row in result["models"]] != expected:
        raise ValueError("The report requires all seven declared model variants")
    for row in result["models"]:
        if row["status"] not in {"evaluated", "unavailable", "failed"}:
            raise ValueError("Unknown model status")
        if row["status"] != "evaluated":
            if row["metrics"] or row["predictions"] or not row.get("reason"):
                raise ValueError("Unevaluated rows need a reason and must not contain scores")
            continue
        for partition in ("validation", "test"):
            saved = row["predictions"][partition]
            truth = result["targets"][partition]
            if len(truth) != len(result["splits"][partition]):
                raise ValueError("Saved target count does not match the partition")
            actual = evaluate_predictions(task, truth, saved["values"],
                probabilities=saved["probabilities"], classes=row.get("classes"),
                positive_label=result.get("positive_label"))
            recorded = row["metrics"][partition]
            # Regression libraries may calculate in float32 before JSON promotes
            # predictions to Python float64. Permit only the resulting rounding error.
            rtol, atol = (1e-6, 1e-7) if task == "regression" else (0, 1e-12)
            if set(actual) != set(recorded) or any(not np.isclose(value, recorded[key], rtol=rtol, atol=atol)
                                                   for key, value in actual.items()):
                raise ValueError(f"{row['name']} {partition} metrics disagree with saved predictions")


def load_reports(root: Path = ROOT) -> list[dict]:
    reports = []
    for slug, title, task, notebook_name, plot in DATASETS:
        directory = root / "results" / slug / "full"
        path = directory / "results.json"
        result = load_results(path)
        verify_result(result, json.loads((root / "datasets" / f"{slug}.json").read_text()), task)
        split_record = json.loads((directory / "splits.json").read_text())
        if (split_record["profile"] != "full" or split_record["seed"] != result["seed"]
                or split_record["splits"] != result["splits"]
                or split_record["source_row_positions"] != result["dataset"]["source_row_positions"]):
            raise ValueError(f"{slug}: split artifact differs from results")
        table = comparison_table(result)
        csv = pd.read_csv(directory / "metrics.csv", index_col=0)
        if csv.index.tolist() != table.index.tolist() or not np.allclose(csv[table.columns], table, equal_nan=True, rtol=0, atol=1e-12):
            raise ValueError(f"{slug}: metric CSV differs from results")
        statuses = {"evaluated": "Evaluated", "unavailable": "Not evaluated", "failed": "Failed"}
        if (csv["status"].tolist() != [statuses[row["status"]] for row in result["models"]]
                or csv["reason"].fillna("").tolist() != [row.get("reason", "") for row in result["models"]]
                or not np.allclose(csv["seconds"], [row["elapsed_seconds"] for row in result["models"]], rtol=0, atol=1e-12)):
            raise ValueError(f"{slug}: CSV statuses, reasons, or timings differ from results")
        manifest = json.loads((root / "checkpoints/manifest.json").read_text())
        if result["checkpoint_manifest"] != manifest:
            raise ValueError(f"{slug}: checkpoint manifest differs from the recorded run")
        for row in result["models"]:
            if row["status"] != "evaluated" or not row["name"].startswith(("TabPFN", "Mitra")):
                continue
            runtime = row["runtime"]
            key = "tabpfn" if row["name"].startswith("TabPFN") else "mitra-regression" if task == "regression" else "mitra-classification"
            if (not runtime["frozen_weights"] or runtime["fine_tuning"]
                    or any(runtime["checkpoint"].get(field) != value for field, value in manifest[key].items())):
                raise ValueError(f"{slug}: foundation-model checkpoint or frozen inference settings differ")
        notebook = nbformat.read(root / "notebooks" / notebook_name, as_version=4)
        nbformat.validate(notebook)
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if not code or any(cell.execution_count is None for cell in code) or any(
            output.output_type == "error" for cell in code for output in cell.outputs
        ):
            raise ValueError(f"{slug}: notebook is unexecuted or contains errors")
        if not (directory / plot).is_file():
            raise ValueError(f"{slug}: primary-metric plot is missing")
        reports.append({"slug": slug, "title": title, "notebook": notebook_name, "plot": plot,
                        "result": result, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return reports


def metric_table(result: dict) -> str:
    table = comparison_table(result)
    lines = ["| Model | " + " | ".join(LABELS[column] for column in table) + " |",
             "| --- | " + " | ".join("---:" for _ in table) + " |"]
    for row in result["models"]:
        cells = [f"{table.loc[row['name'], column]:.4f}" if row["status"] == "evaluated" else
                 "Not evaluated" if row["status"] == "unavailable" else "Failed" for column in table]
        lines.append("| " + row["name"] + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(reports: list[dict]) -> str:
    lines = ["# Tabular benchmark results", "",
        "Compare frozen Mitra v2 and TabPFN-3.5-Fast with validation-tuned XGBoost and LightGBM "
        "on three public tabular datasets. The tables below are generated from saved full-data predictions and metrics; "
        "synthetic demos and 500-row smoke profiles are excluded.", "",
        "## Data and methodology", "",
        "| Dataset | Task | Rows | Source features | Train / validation / test |",
        "| --- | --- | ---: | ---: | --- |"]
    for report in reports:
        r = report["result"]
        source = r["dataset"]["source"]
        features = len(source["numeric_features"]) + len(source.get("categorical_features", [])) + int("date_feature" in source)
        counts = " / ".join(f"{len(r['splits'][key]):,}" for key in ("train", "validation", "test"))
        lines.append(f"| [{report['title']}]({source['source']}) | {r['task']} | {source['rows']:,} | {features} | {counts} |")
    lines += ["", "All datasets are CC BY 4.0; author/citation details, source URLs, and archive/member hashes "
        "are pinned in the [dataset manifests](../datasets/). Targets are Revenue (positive=True), Rented Bike Count, "
        "and Class, respectively, and are excluded from predictors. Every released row is retained.", "",
        "Every model uses the same seed-42 60/20/20 split for its dataset, stratified for classification. "
        "Seoul Bike uses the user-confirmed random-split retrospective tabular prediction protocol with observed weather. "
        "Exploration and all learned preprocessing use training rows only. Validation selects classical settings; "
        "test predictions are produced for each predeclared recipe after that recipe's selection. Earlier test results "
        "do not alter later recipes. There is no train-plus-validation refit, automatic subsampling, resampling, "
        "threshold search, or target transformation.", "",
        "Zero-shot here means frozen pretrained weights with labeled training examples supplied as inference context; "
        "it does not mean inference without labels or without preprocessing. Foundation models use native train-only "
        "preprocessing, no fine-tuning or validation search, TabPFN's auto ensemble, and one Mitra estimator. "
        "Mitra regression uses the distributional head's softmax-weighted bin-center mean. "
        "[Checkpoint identities](../checkpoints/manifest.json) pin the exact task-specific revisions.", "",
        "## Features and classical tuning", "",
        "Raw foundation settings retain source features with only model-required formatting. Seoul Date is parsed "
        "as day/month/year and represented as days since the fixed Unix epoch. Engineered settings retain originals and add:", "",
        "| Dataset | Fixed row-local features |",
        "| --- | --- |",
        "| Online Shoppers | Total pages, total duration, seconds/page, product-page fraction, exit-minus-bounce rate; zero-page ratios become zero |",
        "| Seoul Bike | Hour sine/cosine, weekday, month, weekend indicator, day-of-year sine/cosine |",
        "| Dry Bean | Natural logs of area, perimeter, major-axis length, and minor-axis length |", "",
        "Both classical models use engineered features with train-only imputation, scaling, and categorical encoding "
        "where applicable. Each grid has four candidates with 80 trees: XGBoost learning_rate ∈ {0.05, 0.1}, "
        "max_depth ∈ {3, 6}; LightGBM learning_rate ∈ {0.05, 0.1}, num_leaves ∈ {7, 15}. "
        "Selection maximizes validation ROC-AUC/macro-F1 or minimizes validation RMSE, with first-candidate tie-breaking. "
        "Selected parameters, every trial's validation scores, and preprocessing recipes are saved in each results.json.", "",
        "## Measured test results", "",
        "Values are rounded to four decimals here; JSON/CSV artifacts retain the recorded precision. "
        "ROC-AUC and macro-F1 are primary for classification (higher is better). RMSE is primary for regression "
        "(lower is better); RMSE and MAE use bikes/hour. Average precision is not trapezoidal PR-curve area. "
        "Binary F1/precision/recall use model-default decisions. Dry Bean also exports per-class results and confusion matrices."]
    for report in reports:
        r, slug = report["result"], report["slug"]
        lines += ["", f"### {report['title']}", "", metric_table(r), ""]
        available = comparison_table(r)[r["primary_metric"]].dropna()
        if not available.empty:
            best = available.idxmin() if r["task"] == "regression" else available.idxmax()
            lines.append(f"Best observed test {LABELS[r['primary_metric']]}: **{best}**, **{available[best]:.4f}**. "
                         "This is a descriptive ranking on this split.")
        for row in r["models"]:
            if row["status"] != "evaluated":
                lines += ["", f"**{row['name']}: {'Not evaluated' if row['status'] == 'unavailable' else 'Failed'}** — {row['reason']}"]
        count = sum(row["status"] == "evaluated" for row in r["models"])
        lines += ["", f"Evaluated: {count}/{len(r['models'])}. "
            f"[Executed notebook](../notebooks/{report['notebook']}) · [CSV]({slug}/full/metrics.csv) · "
            f"[Predictions and provenance]({slug}/full/results.json) · [Reproduction guide](../docs/{slug}.md)", "",
            f"![{report['title']} primary-metric comparison]({slug}/full/{report['plot']})", "",
            f"Results artifact SHA-256: `{report['sha256']}`."]
    lines += ["", "## Runtime and environment", "",
        "Elapsed seconds below include worker startup, imports/loading, classical search where applicable, "
        "and validation/test prediction. They are one-run wall times, not isolated training or inference benchmarks. "
        "Classical models ran on CPU with one model thread; foundation models ran on the recorded GPU. "
        "Each model used a separate worker with a 300-second budget and no silent fallback.", "",
        "| Model | " + " | ".join(report["title"] + " (s)" for report in reports) + " |",
        "| --- | " + " | ".join("---:" for _ in reports) + " |"]
    for i, row in enumerate(reports[0]["result"]["models"]):
        name = "Dummy prior / mean" if i == 0 else row["name"]
        lines.append("| " + name + " | " + " | ".join(f"{report['result']['models'][i]['elapsed_seconds']:.1f}" for report in reports) + " |")
    lines += ["", "| Environment | " + " | ".join(report["title"] for report in reports) + " |",
              "| --- | " + " | ".join("---" for _ in reports) + " |"]
    for package in ("python", "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "torch", "tabpfn", "autogluon.tabular"):
        values = [report["result"]["environment"]["python"] if package == "python" else
                  report["result"]["environment"]["packages"].get(package, "Not recorded") for report in reports]
        lines.append("| " + package + " | " + " | ".join(values) + " |")
    devices = sorted({row["runtime"]["cuda_device_name"] for report in reports for row in report["result"]["models"]
                      if row.get("runtime", {}).get("cuda_device_name")})
    lines += ["", "Recorded GPU: " + (", ".join(devices) or "none") + ". "
        "Mitra used bfloat16 autocast on this GPU while checkpoint parameters remained frozen. "
        "Full datasets exceed TabPFN's default CPU row limit; tiny CPU smoke checks do not establish full-data feasibility. "
        "A matching Python/package environment and GPU precision can still produce small cross-device numerical differences.", "",
        "## Interpretation and limitations", "",
        "Foundation-model settings are competitive with these practical, small-grid tree baselines on the recorded splits. "
        "Feature engineering has dataset-dependent effects; compare the separately labeled raw and engineered rows. "
        "The tables do not establish a universally superior family or compare against an exhaustive classical search. "
        "Metrics from different tasks must not be averaged into a common score.", "",
        "These are single-split descriptive results without repeated-seed uncertainty or significance tests. "
        "Online Shoppers retains 125 identical records and session-derived PageValues; availability at a specific live "
        "prediction time is not verified. Seoul Bike includes all 295 non-functioning hours, observed weather, and temporal "
        "neighbors across random partitions; this is not future forecasting. Dry Bean retains 68 identical records and "
        "uses released image-derived measurements, without evaluating image acquisition or segmentation. "
        "Source identifiers are unavailable for further group/independence checks. Unknown overlap with foundation-model "
        "pretraining data is not audited. Conclusions apply to these datasets, splits, and settings.", "",
        "## Reproduction and verification", "",
        "Follow the [README setup and notebook commands](../README.md). Dataset manifests and "
        "[checkpoint revisions](../checkpoints/manifest.json) pin inputs; full JSON records source fingerprints, "
        "row/split positions, predictions, selected settings, and runtime provenance. Smoke profiles and synthetic demos "
        "write separate directories and are not included here.", "", "```bash",
        ".venv/bin/python -m unittest discover -s tests -v",
        ".venv/bin/python scripts/build_report.py",
        ".venv/bin/python scripts/build_report.py --check", "```", "",
        "The report builder requires completed notebooks, full source-row coverage, matching manifests/split/CSV artifacts, "
        "and metrics consistent with saved validation/test predictions (allowing float32 regression rounding). "
        "It does not load model weights, download data, or refit models. `--check` verifies that this report is current "
        "without modifying it. Unevaluated models retain explicit reasons and no substitute scores.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the saved report without writing it")
    args = parser.parse_args()
    content = render_report(load_reports())
    output = ROOT / "results/benchmark_results.md"
    if args.check:
        if not output.exists() or output.read_text() != content:
            parser.exit(1, "Report is stale; run scripts/build_report.py\n")
        print("Verified all three full-data artifacts and current benchmark report")
    else:
        output.write_text(content)
        print(f"Wrote {output.relative_to(ROOT)} from verified full-data results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
