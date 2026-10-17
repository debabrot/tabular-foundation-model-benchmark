"""Real frozen inference, checkpoint provenance, and rejected configurations."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.demo_baseline import demo_fixture
from src.benchmark import load_results, run_comparison, save_results
from src.data import TASKS, make_split
from src.evaluation import comparison_table, evaluate_predictions
from src.models.checkpoints import ROOT, Unavailable, checkpoint
from src.models.tabpfn import FrozenTabPFNClassifier, FrozenTabPFNRegressor, make_tabpfn_models


def weight_digest(modules) -> str:
    digest = hashlib.sha256()
    for module in modules:
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TabPFNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def require_checkpoint(self):
        try:
            return checkpoint("tabpfn", False, {})
        except Unavailable as exc:
            self.skipTest(f"Cached TabPFN weights required for real inference: {exc}")

    def test_real_frozen_weights_match_checkpoint_before_and_after_predictions(self):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        path = self.require_checkpoint()
        reference = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5_FAST, model_path=str(path), device="cpu", n_estimators=1,
        )
        reference.get_inference_config()  # Public API initializes checkpoint modules.
        expected = weight_digest(reference.models_)
        del reference
        X, y = demo_fixture("multiclass")
        split = make_split(y, "multiclass")
        model = FrozenTabPFNClassifier(device="cpu", n_estimators=1)
        with patch("torch.Tensor.backward", side_effect=AssertionError("Backpropagation is forbidden")), patch(
            "torch.optim.AdamW.step", side_effect=AssertionError("Optimizer steps are forbidden"),
        ):
            model.fit(X.iloc[split["train"]], y.iloc[split["train"]])
            self.assertEqual(expected, weight_digest(model.model_.models_))
            predictions = model.predict(X.iloc[split["test"]])
            probabilities = model.predict_proba(X.iloc[split["test"]])
        self.assertEqual(expected, weight_digest(model.model_.models_))
        self.assertTrue(all(not parameter.requires_grad for module in model.model_.models_ for parameter in module.parameters()))
        np.testing.assert_array_equal(predictions, model.classes_[probabilities.argmax(axis=1)])
        np.testing.assert_array_equal(model.classes_, np.unique(y))
        self.assertIsNone(model.model_.tuning_config)
        self.assertFalse(model.model_.differentiable_input)

    def test_all_tasks_use_training_context_and_save_complete_provenance(self):
        self.require_checkpoint()
        for task in TASKS:
            with self.subTest(task=task):
                X, y = demo_fixture(task)
                splits = make_split(y, task)
                models = make_tabpfn_models(task, device="cpu", n_estimators=1)
                cls = FrozenTabPFNRegressor if task == "regression" else FrozenTabPFNClassifier
                original_fit = cls.fit
                seen = []
                def capture_fit(model, features, target):
                    seen.append((features.copy(), target.copy()))
                    return original_fit(model, features, target)
                with patch.object(cls, "fit", capture_fit):
                    results = run_comparison(X, y, models, task=task, dataset_name=task,
                                             dataset_kind="synthetic", splits=splits)
                self.assertEqual(len(seen), 1)
                pd.testing.assert_frame_equal(seen[0][0], X.iloc[splits["train"]])
                pd.testing.assert_series_equal(seen[0][1], y.iloc[splits["train"]])
                row = results["models"][0]
                runtime = row["runtime"]
                self.assertEqual(runtime["model_version"], "v3.5-fast")
                self.assertTrue(runtime["frozen_weights"])
                self.assertFalse(runtime["fine_tuning"])
                self.assertEqual(runtime["device"], "cpu")
                self.assertEqual(runtime["n_estimators_resolved"], 1)
                self.assertEqual(len(runtime["checkpoint"]["revision"]), 40)
                self.assertEqual(len(next(iter(runtime["checkpoint"]["sha256"].values()))), 64)
                self.assertIn("tabpfn", runtime["packages"])
                self.assertIn("torch", runtime["packages"])
                self.assertNotIn("tuning", row)
                self.assertTrue(row["preprocessing"]["estimator"].endswith("FunctionTransformer"))
                self.assertFalse(hasattr(models["TabPFN - Raw"], "runtime_metadata_"))
                for partition in ("validation", "test"):
                    predicted = row["predictions"][partition]
                    self.assertEqual(row["metrics"][partition], evaluate_predictions(
                        task, results["targets"][partition], predicted["values"],
                        probabilities=predicted["probabilities"], classes=row["classes"],
                        positive_label=results["positive_label"],
                    ))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "results.json"
                    save_results(results, path)
                    self.assertEqual(load_results(path), results)
                    self.assertEqual(len(comparison_table(load_results(path))), 1)

    def test_invalid_settings_and_validation_search_are_rejected_before_loading(self):
        X, y = demo_fixture("binary")
        for settings in ({"fine_tune": True}, {"device": "invalid"}, {"n_estimators": 0}, {"n_estimators": True}):
            with self.subTest(settings=settings), patch("src.models.tabpfn.checkpoint") as loader:
                with self.assertRaises(ValueError):
                    FrozenTabPFNClassifier(**settings).fit(X, y)
                loader.assert_not_called()
        with patch("src.models.tabpfn.checkpoint") as loader:
            with self.assertRaisesRegex(ValueError, "validation tuning is disabled"):
                run_comparison(X, y, make_tabpfn_models("binary"), task="binary", dataset_name="invalid",
                               search_spaces={"TabPFN - Raw": {"n_estimators": [1, 2]}})
            loader.assert_not_called()
        with self.assertRaises(ValueError):
            make_tabpfn_models("unknown")

    def test_missing_checkpoint_and_unavailable_cuda_are_actionable(self):
        X, y = demo_fixture("binary")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(Unavailable, "cannot see a CUDA GPU"):
                FrozenTabPFNClassifier(device="cuda").fit(X, y)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            (root / "checkpoints/manifest.json").write_text((ROOT / "checkpoints/manifest.json").read_text())
            with patch("src.models.checkpoints.ROOT", root):
                with self.assertRaisesRegex(Unavailable, "--download"):
                    FrozenTabPFNClassifier(device="cpu").fit(X, y)

    def test_auto_ensemble_uses_checkpoint_default_and_cpu_limit_is_not_bypassed(self):
        from safetensors import safe_open

        path = self.require_checkpoint()
        with safe_open(str(path), framework="pt") as file:
            config = json.loads(file.metadata()["inference_config"])
        X, y = demo_fixture("binary")
        model = FrozenTabPFNClassifier(device="cpu").fit(X.iloc[:90], y.iloc[:90])
        self.assertEqual(model.runtime_metadata_["n_estimators_resolved"], config["N_ESTIMATORS"])
        count = config["MAX_CPU_SAMPLES"] + 1
        with patch.dict("os.environ", {"TABPFN_ALLOW_CPU_LARGE_DATASET": "0"}):
            # Settings are cached by TabPFN; force its actual CPU override off too.
            with patch("tabpfn.settings.settings.tabpfn.allow_cpu_large_dataset", False):
                with self.assertRaisesRegex(Unavailable, "CPU configuration is unsupported"):
                    FrozenTabPFNClassifier(device="cpu", n_estimators=1).fit(
                        pd.DataFrame({"number": np.arange(count)}), np.arange(count) % 2,
                    )

    def test_combined_cli_and_render_only_share_splits_and_keep_artifacts(self):
        self.require_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(ROOT / "scripts/demo_baseline.py"), "--classical", "--tabpfn",
                       "--device", "cpu", "--tabpfn-estimators", "1", "--output-dir", directory]
            environment = {**os.environ, "MPLCONFIGDIR": str(ROOT / ".cache/matplotlib")}
            subprocess.run(command, cwd=directory, env=environment, capture_output=True,
                           text=True, check=True, timeout=120)
            before = {}
            for task in TASKS:
                path = Path(directory) / f"{task}.json"
                before[task] = path.read_bytes()
                results = load_results(path)
                self.assertEqual(len(results["models"]), 5)
                self.assertEqual(len(results["splits"]["train"]), 90)
                row = next(row for row in results["models"] if row["name"] == "TabPFN - Raw")
                self.assertTrue(row["preprocessing"]["estimator"].endswith("FunctionTransformer"))
                self.assertNotIn("tuning", row)
                self.assertTrue((Path(directory) / f"{task}.png").read_bytes().startswith(b"\x89PNG"))
            subprocess.run(command + ["--render-only"], cwd=directory, env=environment,
                           capture_output=True, text=True, check=True, timeout=30)
            for task in TASKS:
                self.assertEqual(before[task], (Path(directory) / f"{task}.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
