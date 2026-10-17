"""Dry Bean source integrity, label alignment, and reproducible feature recipes."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from src.benchmark import load_results, run_comparison, save_results
from src.data import dataset_fingerprint, make_split
from src.datasets import ROOT, load_dry_bean
from src.evaluation import evaluate_predictions
from src.features import BeanFeatures
from src.models.classical import make_classical_models


class DryBeanTests(unittest.TestCase):
    def require_archive(self):
        path = ROOT / ".cache/datasets/dry-bean.zip"
        if not path.exists():
            self.skipTest("Pinned Dry Bean source must be cached; tests never download data")
        return path

    def test_source_labels_features_and_stratified_split(self):
        self.require_archive()
        X, y, metadata = load_dry_bean(download=False)
        self.assertEqual(X.shape, (13611, 16))
        self.assertEqual(X.columns.tolist(), metadata["numeric_features"])
        self.assertEqual(y.name, "Class")
        self.assertNotIn("Class", X)
        self.assertIn("AspectRation", X)
        self.assertIn("roundness", X)
        self.assertEqual(y.value_counts().to_dict(), metadata["class_counts"])
        self.assertTrue(all(isinstance(value, str) for value in y.unique()))
        self.assertEqual(metadata["identical_full_rows"], 68)
        self.assertTrue(np.isfinite(X.to_numpy()).all())
        self.assertTrue((X.to_numpy() > 0).all())
        self.assertEqual(X.iloc[0]["Area"], 28395)
        self.assertEqual(y.iloc[0], "SEKER")
        self.assertEqual(y.iloc[-1], "DERMASON")
        splits = make_split(y, "multiclass")
        self.assertEqual([len(splits[s]) for s in ("train", "validation", "test")], [8166, 2722, 2723])
        for partition, positions in splits.items():
            self.assertEqual(set(y.iloc[positions]), set(metadata["class_counts"]))
            np.testing.assert_array_equal(positions, make_split(y, "multiclass")[partition])
            np.testing.assert_allclose(y.iloc[positions].value_counts(normalize=True).sort_index(),
                                       y.value_counts(normalize=True).sort_index(), atol=0.001, rtol=0)
        again = load_dry_bean(download=False)
        pd.testing.assert_frame_equal(X, again[0])
        pd.testing.assert_series_equal(y, again[1])
        self.assertEqual(dataset_fingerprint(X, y), dataset_fingerprint(*again[:2]))

    def test_missing_corrupt_and_downloaded_cache(self):
        original = self.require_archive().read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_dry_bean(directory, download=False)
            # Exercise first-use download behavior without contacting a service.
            with patch("src.datasets.urlopen", return_value=BytesIO(original)) as download:
                X, _, _ = load_dry_bean(directory, download=True)
                self.assertEqual(len(X), 13611)
                download.assert_called_once()
            archive = Path(directory) / "dry-bean.zip"
            self.assertEqual(archive.read_bytes(), original)
            archive.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "archive checksum mismatch"):
                load_dry_bean(directory, download=False)
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.datasets.urlopen", return_value=BytesIO(b"invalid source")):
                with self.assertRaisesRegex(ValueError, "download checksum"):
                    load_dry_bean(directory, download=True)
            self.assertFalse((Path(directory) / "dry-bean.zip").exists())

    def test_arff_member_hash_is_checked_independently(self):
        original = self.require_archive().read_bytes()
        metadata = json.loads((ROOT / "datasets/dry-bean.json").read_text())
        member = ZipFile(BytesIO(original)).read(metadata["arff_member"])
        with tempfile.TemporaryDirectory() as directory:
            task_root = Path(directory)
            (task_root / "datasets").mkdir()
            archive = task_root / "dry-bean.zip"
            with ZipFile(archive, "w") as source:
                source.writestr(metadata["arff_member"], member + b"\n")
            metadata["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            (task_root / "datasets/dry-bean.json").write_text(json.dumps(metadata))
            with patch("src.datasets.ROOT", task_root):
                with self.assertRaisesRegex(ValueError, "ARFF checksum mismatch"):
                    load_dry_bean(task_root, download=False)

    def test_log_features_are_row_local_and_target_independent(self):
        X = pd.DataFrame({"Area": [1.0, np.e ** 2], "Perimeter": [np.e, np.e ** 3],
                          "MajorAxisLength": [np.e ** 2, np.e], "MinorAxisLength": [1.0, np.e],
                          "Other": [10.0, 20.0]}, index=[30, 10])
        original = X.copy(deep=True)
        features = BeanFeatures().fit(X.iloc[:1], ["unrelated"])
        result = features.transform(X)
        np.testing.assert_allclose(result[list(BeanFeatures.added)], [[0, 1, 2, 0], [2, 3, 1, 1]])
        pd.testing.assert_frame_equal(result[X.columns], X)
        pd.testing.assert_frame_equal(features.transform(X.iloc[[1, 0]]), result.iloc[[1, 0]])
        pd.testing.assert_frame_equal(BeanFeatures().fit(X, ["a", "b"]).transform(X), result)
        pd.testing.assert_frame_equal(X, original)
        for bad in (0.0, -1.0, np.inf, np.nan):
            invalid = X.copy()
            invalid.loc[30, "Area"] = bad
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                features.transform(invalid)
        with self.assertRaisesRegex(ValueError, "already exist"):
            BeanFeatures().fit(result)
        with self.assertRaisesRegex(ValueError, "match fit order"):
            features.transform(X[X.columns[::-1]])

    def test_classical_predictions_keep_original_class_probability_order(self):
        self.require_archive()
        X, y, _ = load_dry_bean(download=False)
        selected, _ = train_test_split(np.arange(len(y)), train_size=500, stratify=y, random_state=42)
        X, y = X.iloc[selected], y.iloc[selected]
        models = {"Dummy prior": DummyClassifier(strategy="prior"), **make_classical_models("multiclass")}
        result = run_comparison(X, y, models, task="multiclass", dataset_name="Dry Bean label fixture")
        for row in result["models"]:
            self.assertEqual(row["classes"], sorted(y.unique()))
            for partition in ("validation", "test"):
                saved = row["predictions"][partition]
                probabilities = np.array(saved["probabilities"])
                np.testing.assert_array_equal(np.array(row["classes"])[probabilities.argmax(axis=1)], saved["values"])
                measured = evaluate_predictions("multiclass", result["targets"][partition], saved["values"],
                    probabilities=probabilities, classes=row["classes"])
                self.assertEqual(row["metrics"][partition], measured)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            save_results(result, path)
            self.assertEqual(load_results(path), result)


if __name__ == "__main__":
    unittest.main()
