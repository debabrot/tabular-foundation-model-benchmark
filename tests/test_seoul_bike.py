"""Pinned source, date compatibility, calendar features, and experiment replay."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.benchmark import load_results, run_comparison, save_results
from src.data import dataset_fingerprint, make_split
from src.datasets import ROOT, load_seoul_bike
from src.evaluation import evaluate_predictions
from src.features import BikeFeatures


class SeoulBikeTests(unittest.TestCase):
    def require_archive(self):
        archive = ROOT / ".cache/datasets/seoul-bike.zip"
        if not archive.exists():
            self.skipTest("Pinned Seoul Bike source must be cached; tests never download data")
        return archive

    def test_source_dates_target_and_all_rows_are_preserved(self):
        self.require_archive()
        X, y, metadata = load_seoul_bike(download=False)
        self.assertEqual(X.shape, (8760, 13))
        self.assertEqual(y.name, "Rented Bike Count")
        self.assertNotIn(y.name, X)
        self.assertFalse(X.isna().any().any())
        self.assertEqual(metadata["non_functioning_rows"], 295)
        self.assertEqual(metadata["identical_full_rows"], 0)
        self.assertEqual(int((X["Functioning Day"] == "No").sum()), 295)
        for column in metadata["categorical_features"]:
            self.assertIsInstance(X[column].dtype, pd.CategoricalDtype)
        dates = pd.to_datetime(X["Date"], unit="D", origin="unix")
        self.assertEqual(str(dates.iloc[0].date()), "2017-12-01")
        self.assertEqual(str(dates.iloc[-1].date()), "2018-11-30")
        # March 1 must not be parsed as January 3.
        self.assertEqual(str(dates.iloc[90 * 24].date()), "2018-03-01")
        stamps = dates + pd.to_timedelta(X["Hour"], unit="h")
        np.testing.assert_array_equal(stamps, pd.date_range("2017-12-01", periods=8760, freq="h"))
        self.assertEqual(y.iloc[0], 254)
        again = load_seoul_bike(download=False)
        self.assertEqual(dataset_fingerprint(X, y), dataset_fingerprint(*again[:2]))
        pd.testing.assert_frame_equal(X, again[0])
        pd.testing.assert_series_equal(y, again[1])

    def test_corrupt_and_missing_cache_are_rejected(self):
        original = self.require_archive().read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_seoul_bike(directory, download=False)
            (Path(directory) / "seoul-bike.zip").write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_seoul_bike(directory, download=False)

    def test_calendar_features_are_row_local_and_target_independent(self):
        dates = pd.to_datetime(["2018-01-01", "2018-01-06", "2020-02-29"])
        X = pd.DataFrame({"Date": (dates - pd.Timestamp("1970-01-01")).days,
                          "Hour": [0, 6, 23], "Weather": [1.0, 2.0, 3.0]}, index=[20, 10, 30])
        original = X.copy(deep=True)
        transformer = BikeFeatures().fit(X.iloc[:1], [999999])
        result = transformer.transform(X)
        self.assertEqual(result.shape, (3, 10))
        np.testing.assert_array_equal(result["DayOfWeek"], [0, 5, 5])
        np.testing.assert_array_equal(result["Month"], [1, 1, 2])
        np.testing.assert_array_equal(result["IsWeekend"], [False, True, True])
        self.assertAlmostEqual(result.loc[20, "HourSin"], 0)
        self.assertAlmostEqual(result.loc[20, "HourCos"], 1)
        self.assertAlmostEqual(result.loc[10, "HourSin"], 1)
        self.assertAlmostEqual(result.loc[20, "DayOfYearCos"], 1)
        self.assertTrue(np.isfinite(result.to_numpy(dtype=float)).all())
        pd.testing.assert_frame_equal(X, original)
        pd.testing.assert_frame_equal(transformer.transform(X.iloc[[2, 0]]), result.iloc[[2, 0]])
        pd.testing.assert_frame_equal(BikeFeatures().fit(X, [-5, 4, 20]).transform(X), result)
        with self.assertRaisesRegex(ValueError, "match fit order"):
            transformer.transform(X[["Hour", "Date", "Weather"]])
        with self.assertRaisesRegex(ValueError, "already exist"):
            BikeFeatures().fit(result)

    def test_full_split_and_baseline_predictions_replay_exactly(self):
        self.require_archive()
        X, y, _ = load_seoul_bike(download=False)
        splits = make_split(y, "regression")
        self.assertEqual([len(splits[s]) for s in ("train", "validation", "test")], [5256, 1752, 1752])
        for key, values in make_split(y, "regression").items():
            np.testing.assert_array_equal(values, splits[key])
        result = run_comparison(X, y, {"Dummy mean": DummyRegressor()}, task="regression",
                                dataset_name="Seoul Bike test", splits=splits)
        row = result["models"][0]
        expected = y.iloc[splits["train"]].mean()
        for partition in ("validation", "test"):
            predictions = row["predictions"][partition]["values"]
            np.testing.assert_array_equal(predictions, np.full(1752, expected))
            measured = evaluate_predictions("regression", y.iloc[splits[partition]], predictions)
            self.assertEqual(row["metrics"][partition], measured)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            save_results(result, path)
            self.assertEqual(load_results(path), result)


if __name__ == "__main__":
    unittest.main()
