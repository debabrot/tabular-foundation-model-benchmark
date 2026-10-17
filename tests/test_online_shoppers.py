"""Dataset provenance, row-local features, and bounded experiment execution."""

from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor

from scripts.demo_baseline import demo_fixture
from src.benchmark import load_results
from src.data import make_split
from src.datasets import ROOT, load_online_shoppers
from src.evaluation import comparison_table
from src.experiments import run_isolated_comparison
from src.features import ShopperFeatures
from src.models.checkpoints import Unavailable


class UnavailableRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        raise Unavailable("Known test prerequisite unavailable")


class SlowRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        time.sleep(60)
        return self


class BrokenRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        raise ValueError("Unexpected test implementation failure")


class OnlineShoppersTests(unittest.TestCase):
    def require_archive(self):
        path = ROOT / ".cache/datasets/online-shoppers.zip"
        if not path.exists():
            self.skipTest("Pinned dataset must be cached; tests do not download it")
        return path

    def test_pinned_dataset_schema_categories_and_split(self):
        self.require_archive()
        X, y, metadata = load_online_shoppers(download=False)
        self.assertEqual(X.shape, (12330, 17))
        self.assertNotIn("Revenue", X)
        self.assertEqual(y.name, "Revenue")
        self.assertEqual(int(y.sum()), 1908)
        self.assertFalse(X.isna().any().any())
        for name in metadata["categorical_features"]:
            self.assertIsInstance(X[name].dtype, pd.CategoricalDtype)
        self.assertEqual(metadata["identical_full_rows"], 125)
        splits = make_split(y, "binary")
        self.assertEqual([len(splits[key]) for key in ("train", "validation", "test")], [7398, 2466, 2466])
        for key in splits:
            self.assertAlmostEqual(y.iloc[splits[key]].mean(), y.mean(), delta=0.001)
        again = load_online_shoppers(download=False)
        pd.testing.assert_frame_equal(X, again[0])
        pd.testing.assert_series_equal(y, again[1])

    def test_corrupt_cache_is_rejected(self):
        original = self.require_archive().read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "online-shoppers.zip").write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_online_shoppers(directory, download=False)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_online_shoppers(directory, download=False)

    def test_engineered_features_are_row_local_and_target_independent(self):
        X = pd.DataFrame({"Administrative": [3, 0], "Informational": [2, 0], "ProductRelated": [5, 0],
                          "Administrative_Duration": [30, 0], "Informational_Duration": [20, 0],
                          "ProductRelated_Duration": [50, 0], "ExitRates": [0.03, 0], "BounceRates": [0.01, 0]})
        original = X.copy(deep=True)
        features = ShopperFeatures().fit(X.iloc[:1], [True])
        result = features.transform(X)
        self.assertEqual(result.loc[0, "TotalPages"], 10)
        self.assertEqual(result.loc[0, "TotalDuration"], 100)
        self.assertEqual(result.loc[0, "SecondsPerPage"], 10)
        self.assertEqual(result.loc[0, "ProductPageFraction"], 0.5)
        self.assertAlmostEqual(result.loc[0, "ExitBounceGap"], 0.02)
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertEqual(result.loc[1, "SecondsPerPage"], 0)
        pd.testing.assert_frame_equal(X, original)
        pd.testing.assert_frame_equal(features.transform(X.iloc[[1]]), result.iloc[[1]])
        pd.testing.assert_frame_equal(ShopperFeatures().fit(X, [False, False]).transform(X), result)

    def test_isolated_runs_preserve_splits_and_distinguish_unavailable_and_failed(self):
        X, y = demo_fixture("regression")
        splits = make_split(y, "regression")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            result = run_isolated_comparison(X, y, {
                "Mean": DummyRegressor(), "Unavailable": UnavailableRegressor(), "Broken": BrokenRegressor(),
            }, task="regression", dataset_name="isolation fixture", splits=splits, timeout=30, output_path=path)
            self.assertEqual([row["status"] for row in result["models"]], ["evaluated", "unavailable", "failed"])
            for key in splits:
                np.testing.assert_array_equal(result["splits"][key], splits[key])
            for row in result["models"][1:]:
                self.assertFalse(row["metrics"])
                self.assertFalse(row["predictions"])
            self.assertTrue(comparison_table(result).loc["Unavailable"].isna().all())
            self.assertTrue(comparison_table(result).loc["Broken"].isna().all())
            self.assertEqual(load_results(path), result)
            expected = np.mean(y.iloc[splits["train"]])
            np.testing.assert_array_equal(result["models"][0]["predictions"]["test"]["values"], expected)

    def test_timeout_never_generates_metrics(self):
        X, y = demo_fixture("regression")
        result = run_isolated_comparison(X, y, {"Slow": SlowRegressor()}, task="regression",
                                         dataset_name="timeout fixture", timeout=0.5)
        self.assertEqual(result["models"][0]["status"], "unavailable")
        self.assertIn("runtime budget", result["models"][0]["reason"])
        self.assertTrue(comparison_table(result).isna().all().all())


if __name__ == "__main__":
    unittest.main()
