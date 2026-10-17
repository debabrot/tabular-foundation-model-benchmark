"""Verify experimental isolation and the baseline workflow end to end."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier

from scripts.demo_baseline import demo_fixture
from src.benchmark import load_results, run_comparison, save_results
from src.data import TASKS, dataset_fingerprint, make_split, validate_split
from src.evaluation import comparison_table, evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]


class RecordingRegressor(RegressorMixin, BaseEstimator):
    fits = []
    queries = []

    def fit(self, X, y):
        self.mean_ = np.mean(y)
        type(self).fits.append((np.array(X), np.array(y)))
        return self

    def predict(self, X):
        type(self).queries.append(np.array(X))
        return np.full(len(X), self.mean_)


class BaselineTests(unittest.TestCase):
    def test_splits_are_reproducible_stratified_and_disjoint(self):
        for task in TASKS:
            X, y = demo_fixture(task)
            first, repeated = make_split(y, task), make_split(y, task)
            for key in first:
                np.testing.assert_array_equal(first[key], repeated[key])
            self.assertEqual([len(first[k]) for k in ("train", "validation", "test")], [90, 30, 30])
            np.testing.assert_array_equal(np.sort(np.concatenate(list(first.values()))), np.arange(len(X)))
            self.assertFalse(np.array_equal(first["train"], make_split(y, task, seed=7)["train"]))
            if task == "binary":
                for positions in first.values():
                    self.assertAlmostEqual(y.iloc[positions].mean(), y.mean(), delta=0.04)
            invalid = {**first, "test": first["validation"]}
            with self.assertRaises(ValueError):
                validate_split(invalid, y.to_numpy(), task)

    def test_default_preprocessing_and_estimator_fit_only_training_rows(self):
        splits = {"train": np.arange(90), "validation": np.arange(90, 120), "test": np.arange(120, 150)}
        X = pd.DataFrame({"numeric": np.arange(150, dtype=float), "category": ["train"] * 90 + ["unseen"] * 60})
        X.loc[90:, "numeric"] = 1e8
        X.loc[0, "numeric"] = np.nan
        X.loc[1, "category"] = None
        y = pd.Series(np.arange(150, dtype=float))
        y.loc[90:] = 1e9
        RecordingRegressor.fits, RecordingRegressor.queries = [], []
        original = RecordingRegressor()
        results = run_comparison(X, y, {"Recorder": original}, task="regression", dataset_name="isolation", splits=splits)
        fitted_X, fitted_y = RecordingRegressor.fits[0]
        self.assertEqual(fitted_X.shape, (90, 2))  # The held-out category was never learned.
        self.assertAlmostEqual(fitted_X[:, 0].mean(), 0, places=12)
        np.testing.assert_array_equal(fitted_y, y.iloc[:90])
        for query in RecordingRegressor.queries:
            self.assertTrue((query[:, 0] > 1e6).all())  # Scaling statistics came from train.
            np.testing.assert_array_equal(query[:, 1], 0)  # Unknown category is ignored.
        self.assertFalse(hasattr(original, "mean_"))  # Caller estimator was cloned.
        self.assertEqual(results["models"][0]["predictions"]["test"]["values"], [44.5] * 30)

    def test_binary_metrics_use_explicit_probability_class_order(self):
        truth, labels = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
        positive_scores = np.array([0.1, 0.4, 0.35, 0.8])
        metrics = evaluate_predictions(
            "binary", truth, labels, classes=np.array([1, 0]), positive_label=1,
            probabilities=np.column_stack([positive_scores, 1 - positive_scores]),
        )
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["average_precision"], 5 / 6)
        for key in ("f1", "precision", "recall", "accuracy"):
            self.assertEqual(metrics[key], 0.5)
        with self.assertRaises(ValueError):
            evaluate_predictions("binary", truth, labels, classes=[0, 1], positive_label=1,
                                 probabilities=np.ones((4, 2)))

    def test_multiclass_and_regression_metrics_match_hand_calculation(self):
        metrics = evaluate_predictions("multiclass", ["a", "a", "b", "c"], ["a", "b", "b", "b"], classes=["a", "b", "c"])
        self.assertAlmostEqual(metrics["macro_f1"], 7 / 18)
        self.assertAlmostEqual(metrics["weighted_f1"], 11 / 24)
        self.assertEqual(metrics["accuracy"], 0.5)
        metrics = evaluate_predictions("regression", [1, 2, 3], [1, 2, 5])
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(4 / 3))
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertEqual(metrics["r2"], -1)

    def test_saved_results_preserve_predictions_splits_and_metrics(self):
        X, y = demo_fixture("binary")
        models = {"Prior": DummyClassifier(strategy="prior"), "Stratified": DummyClassifier(strategy="stratified")}
        result = run_comparison(X, y, models, task="binary", dataset_name="binary", dataset_kind="synthetic")
        repeated = run_comparison(X, y, models, task="binary", dataset_name="binary", dataset_kind="synthetic")
        self.assertEqual(result, repeated)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_results(result, path)
            loaded = load_results(path)
            self.assertEqual(result, loaded)
        for row in loaded["models"]:
            for partition in ("validation", "test"):
                np.testing.assert_array_equal(loaded["targets"][partition], y.iloc[loaded["splits"][partition]])
                predicted = row["predictions"][partition]
                self.assertEqual(row["metrics"][partition], evaluate_predictions(
                    "binary", loaded["targets"][partition], predicted["values"],
                    probabilities=predicted["probabilities"], classes=row["classes"],
                    positive_label=loaded["positive_label"],
                ))
        self.assertEqual(loaded["dataset"]["fingerprint_sha256"], dataset_fingerprint(X, y))
        changed = X.copy()
        changed.loc[0, "feature_1"] += 1
        self.assertNotEqual(dataset_fingerprint(X, y), dataset_fingerprint(changed, y))
        with self.assertRaises(ValueError):
            run_comparison(X, y.iloc[::-1], models, task="binary", dataset_name="misaligned")

    def test_demo_runs_from_another_directory_and_renders_without_refitting(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(ROOT / "scripts/demo_baseline.py"), "--output-dir", directory]
            env = {**os.environ, "MPLCONFIGDIR": str(ROOT / ".cache/matplotlib")}
            subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, check=True, timeout=60)
            before = {}
            for task in TASKS:
                root = Path(directory)
                report = root / f"{task}.json"
                before[task] = report.read_bytes()
                loaded = load_results(report)
                self.assertEqual(loaded["dataset"]["kind"], "synthetic")
                self.assertTrue((root / f"{task}.png").read_bytes().startswith(b"\x89PNG"))
                pd.testing.assert_frame_equal(
                    pd.read_csv(root / f"{task}.csv", index_col="model"), comparison_table(loaded),
                )
                (root / f"{task}.csv").unlink()
                (root / f"{task}.png").unlink()
            subprocess.run(command + ["--render-only"], cwd=directory, env=env, capture_output=True,
                           text=True, check=True, timeout=60)
            for task in TASKS:
                self.assertEqual(before[task], (Path(directory) / f"{task}.json").read_bytes())
                self.assertTrue((Path(directory) / f"{task}.csv").exists())
                self.assertTrue((Path(directory) / f"{task}.png").exists())
            # Independent process fits produce the same JSON, including provenance settings.
            subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, check=True, timeout=60)
            for task in TASKS:
                self.assertEqual(before[task], (Path(directory) / f"{task}.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
