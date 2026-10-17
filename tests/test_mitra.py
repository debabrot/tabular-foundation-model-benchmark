"""Validate actual frozen Mitra v2 weights, mean decoding, and train-only context."""

import hashlib
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
from src.models.mitra import FrozenMitraClassifier, FrozenMitraRegressor, make_mitra_models, regression_mean


def state_digest(state) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class MitraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def require_checkpoint(self, regression=False):
        try:
            return checkpoint("mitra-regression" if regression else "mitra-classification", False, {})
        except Unavailable as exc:
            self.skipTest(f"Cached Mitra v2 weights required: {exc}")

    def test_mean_decode_and_original_target_units(self):
        logits = torch.log(torch.tensor([[0.25, 0.75]]))
        result = regression_mean(logits, torch.tensor([-0.5, 0.5, 1.5]))
        self.assertAlmostEqual(result.item(), 0.75, places=6)
        self.assertNotEqual(result.item(), 1.0)  # Most likely bin would be wrong.

    def test_both_checkpoints_remain_frozen_without_trainer_or_optimizer(self):
        from safetensors.torch import load_file

        for task in ("binary", "regression"):
            with self.subTest(task=task):
                path = self.require_checkpoint(task == "regression")
                expected = state_digest(load_file(str(path / "model.safetensors")))
                X, y = demo_fixture(task)
                original = X.copy(deep=True)
                cls = FrozenMitraRegressor if task == "regression" else FrozenMitraClassifier
                np_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
                with patch("torch.optim.AdamW", side_effect=AssertionError("Optimizer forbidden")), patch(
                    "torch.Tensor.backward", side_effect=AssertionError("Backpropagation forbidden"),
                ), patch("autogluon.tabular.models.mitra._internal.core.trainer_finetune.TrainerFinetune.__init__",
                         side_effect=AssertionError("Trainer forbidden")):
                    model = cls(device="cpu").fit(X.iloc[:90], y.iloc[:90])
                    self.assertEqual(model.runtime_metadata_["precision"], "float32")
                    self.assertEqual(expected, state_digest(model.model_.state_dict()))
                    predicted = model.predict(X.iloc[90:])
                self.assertEqual(expected, state_digest(model.model_.state_dict()))
                self.assertTrue(all(not parameter.requires_grad for parameter in model.model_.parameters()))
                self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
                np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
                self.assertEqual(np_state[2:], np.random.get_state()[2:])
                pd.testing.assert_frame_equal(X, original)
                self.assertEqual(predicted.shape, (60,))
                if task == "regression":
                    self.assertEqual(model.model_.dim_output, 1000)
                    seen = []
                    hook = model.model_.register_forward_hook(lambda module, args, output: seen.append(output.detach().cpu().numpy()[0]))
                    predicted = model.predict(X.iloc[90:93])
                    hook.remove()
                    raw = seen[0]
                    probability = np.exp(raw - raw.max(axis=1, keepdims=True))
                    probability /= probability.sum(axis=1, keepdims=True)
                    edges = np.linspace(-0.5, 1.5, 1001)
                    normalized = probability @ ((edges[:-1] + edges[1:]) / 2)
                    if model.runtime_metadata_["target_mirror"]:
                        normalized = 1 - normalized
                    metadata = model.runtime_metadata_
                    expected_prediction = normalized * (metadata["target_max"] - metadata["target_min"]) + metadata["target_min"]
                    np.testing.assert_allclose(predicted, expected_prediction, rtol=1e-4, atol=1e-4)

    def test_all_tasks_use_only_training_context_and_persist_metrics(self):
        for task in TASKS:
            with self.subTest(task=task):
                self.require_checkpoint(task == "regression")
                X, y = demo_fixture(task)
                splits = make_split(y, task)
                cls = FrozenMitraRegressor if task == "regression" else FrozenMitraClassifier
                original_fit, seen = cls.fit, []
                def capture(model, features, target):
                    seen.append((features.copy(), target.copy()))
                    return original_fit(model, features, target)
                models = make_mitra_models(task, device="cpu")
                with patch.object(cls, "fit", capture):
                    results = run_comparison(X, y, models, task=task, dataset_name=task, splits=splits)
                self.assertEqual(len(seen), 1)
                pd.testing.assert_frame_equal(seen[0][0], X.iloc[splits["train"]])
                pd.testing.assert_series_equal(seen[0][1], y.iloc[splits["train"]])
                row = results["models"][0]
                self.assertEqual(row["runtime"]["support_rows"], 90)
                self.assertEqual(row["runtime"]["support_sampling"], "none")
                self.assertEqual(row["runtime"]["model_version"], "mitra-v2")
                self.assertEqual(len(row["runtime"]["checkpoint"]["revision"]), 40)
                self.assertTrue(all(len(value) == 64 for value in row["runtime"]["checkpoint"]["sha256"].values()))
                self.assertNotIn("tuning", row)
                self.assertFalse(hasattr(models["Mitra v2 - Raw"], "runtime_metadata_"))
                for partition in ("validation", "test"):
                    saved = row["predictions"][partition]
                    self.assertEqual(row["metrics"][partition], evaluate_predictions(
                        task, results["targets"][partition], saved["values"], probabilities=saved["probabilities"],
                        classes=row["classes"], positive_label=results["positive_label"],
                    ))
                    if task != "regression":
                        probabilities = np.array(saved["probabilities"])
                        np.testing.assert_array_equal(saved["values"], np.array(row["classes"])[probabilities.argmax(axis=1)])
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "result.json"
                    save_results(results, path)
                    self.assertEqual(load_results(path), results)
                    self.assertEqual(len(comparison_table(load_results(path))), 1)

    def test_query_batching_and_repeated_calls_are_reproducible(self):
        self.require_checkpoint()
        X, y = demo_fixture("multiclass")
        model = FrozenMitraClassifier(device="cpu", batch_size=7).fit(X.iloc[:90], y.iloc[:90])
        first = model.predict_proba(X.iloc[90:111])
        np.testing.assert_array_equal(first, model.predict_proba(X.iloc[90:111]))
        model.batch_size = 256
        np.testing.assert_allclose(first, model.predict_proba(X.iloc[90:111]), atol=1e-5)
        np.testing.assert_array_equal(model.classes_, np.unique(y))
        with self.assertRaisesRegex(ValueError, "columns must match"):
            model.predict(X[X.columns[::-1]].iloc[90:])

    def test_invalid_configurations_and_tuning_do_not_load_weights(self):
        X, y = demo_fixture("binary")
        cases = ({"fine_tune": True}, {"device": "bad"}, {"batch_size": 0}, {"random_state": None})
        for settings in cases:
            with self.subTest(settings=settings), patch("src.models.mitra.checkpoint") as loader:
                with self.assertRaises(ValueError):
                    FrozenMitraClassifier(**settings).fit(X, y)
                loader.assert_not_called()
        with patch("src.models.mitra.checkpoint") as loader:
            with self.assertRaisesRegex(ValueError, "validation tuning is disabled"):
                run_comparison(X, y, make_mitra_models("binary"), task="binary", dataset_name="invalid",
                               search_spaces={"Mitra v2 - Raw": {"batch_size": [7, 32]}})
            with self.assertRaisesRegex(Unavailable, "no rows or features were sampled"):
                FrozenMitraClassifier(max_support_rows=10).fit(X, y)
            with self.assertRaisesRegex(Unavailable, "2–10 classes"):
                FrozenMitraClassifier().fit(X, np.arange(len(X)) % 11)
            loader.assert_not_called()
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(Unavailable, "cannot see a CUDA GPU"):
                FrozenMitraClassifier(device="cuda").fit(X, y)
        with self.assertRaisesRegex(ValueError, "varying training targets"):
            FrozenMitraRegressor().fit(X, np.zeros(len(X)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            (root / "checkpoints/manifest.json").write_text((ROOT / "checkpoints/manifest.json").read_text())
            with patch("src.models.checkpoints.ROOT", root):
                with self.assertRaisesRegex(Unavailable, "--download"):
                    FrozenMitraClassifier(device="cpu").fit(X, y)

    def test_mitra_cli_reproduces_and_renders_saved_outputs(self):
        self.require_checkpoint()
        self.require_checkpoint(regression=True)
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(ROOT / "scripts/demo_baseline.py"), "--mitra", "--device", "cpu",
                       "--output-dir", directory]
            env = {**os.environ, "MPLCONFIGDIR": str(ROOT / ".cache/matplotlib")}
            def run(extra=()):
                subprocess.run(command + list(extra), cwd=directory, env=env, capture_output=True,
                               text=True, check=True, timeout=90)
                return {task: (Path(directory) / f"{task}.json").read_bytes() for task in TASKS}
            first = run()
            self.assertEqual(first, run())
            self.assertEqual(first, run(["--render-only"]))
            for task in TASKS:
                result = load_results(Path(directory) / f"{task}.json")
                self.assertEqual(len(result["models"]), 3)
                self.assertTrue((Path(directory) / f"{task}.png").read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
