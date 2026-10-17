"""Check real tree inference and test-set isolation during validation selection."""

import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import FunctionTransformer

from scripts.demo_baseline import demo_fixture
from src.benchmark import load_results, run_comparison
from src.data import TASKS, make_split
from src.evaluation import comparison_table
from src.models.classical import EncodedXGBClassifier, classical_search_spaces, make_classical_models

ROOT = Path(__file__).resolve().parents[1]


class AuditRegressor(RegressorMixin, BaseEstimator):
    events = []

    def __init__(self, value=0):
        self.value = value

    def fit(self, X, y):
        type(self).events.append(("fit", X.index.tolist(), np.asarray(y).tolist()))
        self.fitted_ = True
        return self

    def predict(self, X):
        type(self).events.append(("predict", X.index.tolist()))
        return np.full(len(X), self.value, dtype=float)


def without_timings(results: dict) -> dict:
    results = copy.deepcopy(results)
    for row in results["models"]:
        for trial in row.get("tuning", {}).get("trials", []):
            trial.pop("fit_and_validation_seconds")
    return results


class ClassicalTests(unittest.TestCase):
    def test_both_families_tune_and_predict_for_all_tasks(self):
        for task in TASKS:
            with self.subTest(task=task):
                X, y = demo_fixture(task)
                models = make_classical_models(task)
                result = run_comparison(X, y, models, task=task, dataset_name=task,
                                        search_spaces=classical_search_spaces())
                for row in result["models"]:
                    self.assertEqual(row["tuning"]["candidate_count"], 4)
                    trials = row["tuning"]["trials"]
                    scores = [trial["validation_metrics"][result["primary_metric"]] for trial in trials]
                    best = min(scores) if task == "regression" else max(scores)
                    selected = row["tuning"]["selected_index"]
                    self.assertEqual(selected, scores.index(best))
                    self.assertEqual(row["metrics"]["validation"][result["primary_metric"]], best)
                    self.assertTrue(all(trial["fit_and_validation_seconds"] >= 0 for trial in trials))
                    for key, value in trials[selected]["parameters"].items():
                        self.assertEqual(row["parameters"][key], value)
                    if task != "regression":
                        np.testing.assert_array_equal(row["classes"], np.unique(y))
                        proba = np.array(row["predictions"]["test"]["probabilities"])
                        np.testing.assert_array_equal(
                            row["predictions"]["test"]["values"], np.asarray(row["classes"])[proba.argmax(axis=1)],
                        )
                    self.assertFalse(hasattr(models[row["name"]], "n_features_in_"))
                for package in ("xgboost", "lightgbm"):
                    self.assertIn(package, result["environment"]["packages"])

    def test_xgboost_preserves_nonconsecutive_numeric_labels(self):
        X, y = demo_fixture("binary")
        features = X.select_dtypes("number").fillna(0)
        labels = np.where(y == 1, 30, 10)
        model = EncodedXGBClassifier(n_estimators=5).fit(features, labels)
        np.testing.assert_array_equal(model.classes_, [10, 30])
        probabilities = model.predict_proba(features)
        np.testing.assert_array_equal(model.predict(features), model.classes_[probabilities.argmax(axis=1)])
        np.testing.assert_array_equal(probabilities, model.model_.predict_proba(features))

    def test_all_selection_finishes_before_test_and_ignores_test_targets(self):
        X = pd.DataFrame({"value": np.arange(30, dtype=float)})
        y = pd.Series([0.0] * 18 + [7.0] * 6 + [3.0] * 6)
        splits = {"train": list(range(18)), "validation": list(range(18, 24)), "test": list(range(24, 30))}
        models = {"First": AuditRegressor(), "Second": AuditRegressor()}
        spaces = {name: {"value": [3, 7]} for name in models}
        kwargs = dict(task="regression", dataset_name="selection-isolation", splits=splits,
                      preprocessor=FunctionTransformer(), search_spaces=spaces)
        AuditRegressor.events = []
        result = run_comparison(X, y, models, **kwargs)
        events = AuditRegressor.events
        first_test = next(i for i, event in enumerate(events) if event[1] == splits["test"])
        self.assertEqual(len([event for event in events[:first_test] if event[0] == "fit"]), 4)
        self.assertEqual(len(events[first_test:]), 2)  # One final test prediction per selected model.
        for event in events:
            if event[0] == "fit":
                self.assertEqual(event[1], splits["train"])
                self.assertEqual(event[2], [0.0] * 18)
        changed = y.copy()
        changed.iloc[splits["test"]] = -1000
        second = run_comparison(X, changed, models, **kwargs)
        for first_row, second_row in zip(result["models"], second["models"]):
            self.assertEqual(first_row["parameters"]["value"], 7)  # Test-optimal value 3 must lose.
            self.assertEqual(first_row["parameters"], second_row["parameters"])
            self.assertEqual(first_row["metrics"]["validation"], second_row["metrics"]["validation"])
            self.assertNotEqual(first_row["metrics"]["test"], second_row["metrics"]["test"])
        self.assertTrue(all(model.value == 0 for model in models.values()))
        # Equal validation errors resolve by candidate order, never test performance.
        tied = run_comparison(X, y, {"Tie": AuditRegressor()}, **{
            **kwargs, "search_spaces": {"Tie": {"value": [5, 9]}},
        })
        self.assertEqual(tied["models"][0]["parameters"]["value"], 5)

    def test_changing_test_features_cannot_change_selection_or_preprocessing(self):
        X, y = demo_fixture("regression")
        splits = make_split(y, "regression")
        kwargs = dict(task="regression", dataset_name="feature-isolation", splits=splits,
                      search_spaces=classical_search_spaces())
        before = run_comparison(X, y, make_classical_models("regression"), **kwargs)
        changed = X.copy()
        changed.iloc[splits["test"], 0] = 1e12
        changed.iloc[splits["test"], changed.columns.get_loc("category")] = "test-only"
        after = run_comparison(changed, y, make_classical_models("regression"), **kwargs)
        for original, modified in zip(before["models"], after["models"]):
            self.assertEqual(original["parameters"], modified["parameters"])
            self.assertEqual(original["predictions"]["validation"], modified["predictions"]["validation"])

    def test_classical_demo_repeats_and_renders_saved_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(ROOT / "scripts/demo_baseline.py"), "--classical", "--output-dir", directory]
            env = {**os.environ, "MPLCONFIGDIR": str(ROOT / ".cache/matplotlib")}
            def run(extra=()):
                subprocess.run(command + list(extra), cwd=directory, env=env, capture_output=True,
                               text=True, check=True, timeout=60)
                return {task: load_results(Path(directory) / f"{task}.json") for task in TASKS}
            first, repeated = run(), run()
            for task in TASKS:
                self.assertEqual(without_timings(first[task]), without_timings(repeated[task]))
                self.assertEqual(len(first[task]["models"]), 4)
            rendered = run(["--render-only"])
            self.assertEqual(repeated, rendered)
            for task in TASKS:
                pd.testing.assert_frame_equal(
                    pd.read_csv(Path(directory) / f"{task}.csv", index_col="model"),
                    comparison_table(rendered[task]),
                )
                self.assertTrue((Path(directory) / f"{task}.png").read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
