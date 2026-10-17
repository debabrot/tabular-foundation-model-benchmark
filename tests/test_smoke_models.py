"""Verify the runtime probe's data isolation, validation, and command contract."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.smoke_models import ROOT, TASKS, fixture, validate_predictions


class SmokeModelsTests(unittest.TestCase):
    def test_hub_symlink_preserves_safetensors_format_after_resolution(self):
        import torch
        from safetensors.torch import save_file
        from tabpfn.checkpoint import Checkpoint
        from src.models.checkpoints import checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            (root / "checkpoints/manifest.json").write_text((ROOT / "checkpoints/manifest.json").read_text())
            blob = root / "extensionless-blob"
            save_file({"weight": torch.zeros(1)}, str(blob))
            link = root / "model.safetensors"
            link.symlink_to(blob)
            with patch("src.models.checkpoints.ROOT", root), patch(
                "huggingface_hub.hf_hub_download", return_value=str(link),
            ):
                path = checkpoint("tabpfn", False, {})
            # The installed TabPFN loader resolves the path before detecting format.
            loaded = Checkpoint(path.resolve()).load()
            self.assertIn("weight", loaded["state_dict"])

    def test_fixtures_repeat_without_overlapping_partitions(self):
        for task in TASKS:
            with self.subTest(task=task):
                first, second = fixture(task), fixture(task)
                for left, right in zip(first, second):
                    np.testing.assert_array_equal(left, right)
                train, valid, test, y_train, y_valid = first
                self.assertEqual((len(train), len(valid), len(test)), (54, 18, 18))
                row_sets = [set(map(tuple, rows)) for rows in (train, valid, test)]
                self.assertFalse(row_sets[0] & row_sets[1])
                self.assertFalse(row_sets[0] & row_sets[2])
                self.assertFalse(row_sets[1] & row_sets[2])
                if task != "regression":
                    self.assertEqual(len(np.unique(y_train)), 2 if task == "binary" else 3)
                    np.testing.assert_array_equal(np.unique(y_train), np.unique(y_valid))

    def test_rejects_invalid_predictions_and_probabilities(self):
        valid = np.tile([0.25, 0.75], (18, 1))
        cases = [
            (np.zeros(17), valid, [0, 1]),
            (np.full(18, np.nan), valid, [0, 1]),
            (np.full(18, 2), valid, [0, 1]),
            (np.zeros(18), valid, [1, 0]),
            (np.zeros(18), np.ones((18, 2)), [0, 1]),
            (np.zeros(18), np.tile([-0.1, 1.1], (18, 1)), [0, 1]),
        ]
        for predictions, probabilities, classes in cases:
            with self.subTest(predictions=predictions.shape, classes=classes):
                with self.assertRaises(ValueError):
                    validate_predictions(predictions, probabilities, "binary", classes)

    def run_probe(self, directory, *options):
        output = Path(directory) / "report.json"
        process = subprocess.run(
            [sys.executable, str(ROOT / "scripts/smoke_models.py"),
             "--output", str(output), *options],
            cwd=directory, capture_output=True, text=True, timeout=90,
        )
        self.assertTrue(output.exists(), process.stderr)
        return process.returncode, json.loads(output.read_text())

    def test_classical_models_predict_for_all_task_types(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.run_probe(directory, "--models", "xgboost", "lightgbm")
            self.assertEqual(code, 0, report)
            self.assertEqual(len(report["checks"]), 6)
            self.assertTrue(all(check["status"] == "passed" for check in report["checks"]))
            # A second independent process reproduces the first model's predictions.
            _, repeated = self.run_probe(directory, "--models", "xgboost", "--tasks", "binary")
            self.assertEqual(report["checks"][0]["predictions"], repeated["checks"][0]["predictions"])

    def test_mitra_v2_regression_runs_without_fine_tuning(self):
        from src.models.checkpoints import checkpoint, Unavailable

        try:
            checkpoint("mitra-regression", False, {})
        except Unavailable as exc:
            self.skipTest(f"Cached Mitra v2 regression checkpoint required: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.run_probe(directory, "--models", "mitra", "--tasks", "regression")
            self.assertEqual(code, 0, report)
            self.assertEqual(report["checks"][0]["status"], "passed")
            self.assertEqual(len(report["checks"][0]["predictions"]), 18)
            self.assertEqual(report["checks"][0]["checkpoint"]["repo_id"], "autogluon/mitra-regressor-2")
            self.assertEqual(report["checks"][0]["runtime"]["architecture"]["dim_output"], 1000)
            self.assertFalse(report["checks"][0]["runtime"]["fine_tuning"])


if __name__ == "__main__":
    unittest.main()
