"""Check model inference on tiny synthetic fixtures, without benchmark datasets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.checkpoints import Unavailable, checkpoint
MODELS = ("xgboost", "lightgbm", "tabpfn", "mitra")
TASKS = ("binary", "multiclass", "regression")
SEED = 42


def fixture(task: str):
    import numpy as np
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    if task == "regression":
        X, y = make_regression(n_samples=90, n_features=4, random_state=SEED)
    else:
        X, y = make_classification(
            n_samples=90, n_features=4, n_informative=3, n_redundant=0,
            n_classes=2 if task == "binary" else 3, n_clusters_per_class=1,
            random_state=SEED,
        )
    X = X.astype(np.float32)
    stratify = y if task != "regression" else None
    # 54 training and 36 held-out observations; then 18 validation and 18 test.
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=stratify,
    )
    X_valid, X_test, y_valid, _ = train_test_split(
        X_hold, y_hold, test_size=0.5, random_state=SEED,
        stratify=y_hold if stratify is not None else None,
    )
    return X_train, X_valid, X_test, y_train, y_valid


def make_model(name: str, task: str, args, record: dict):
    regression = task == "regression"
    if name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        settings = dict(n_estimators=5, max_depth=2, n_jobs=1, random_state=SEED, device="cpu")
        record["settings"] = settings
        return (XGBRegressor if regression else XGBClassifier)(**settings)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        settings = dict(n_estimators=5, max_depth=2, num_leaves=4, min_child_samples=2,
                        n_jobs=1, random_state=SEED, verbosity=-1, deterministic=True,
                        force_col_wise=True, device_type="cpu")
        record["settings"] = settings
        return (LGBMRegressor if regression else LGBMClassifier)(**settings)
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        path = checkpoint("tabpfn", args.download, record)
        settings = dict(device=record["device"], n_estimators=1, random_state=SEED,
                        n_preprocessing_jobs=1)
        record["settings"] = {**settings, "version": "v3.5-fast", "fine_tuning": False}
        cls = TabPFNRegressor if regression else TabPFNClassifier
        return cls.create_default_for_version(ModelVersion.V3_5_FAST, model_path=str(path), **settings)
    from src.models.mitra import make_mitra_models

    model = next(iter(make_mitra_models(task, SEED, record["device"], args.download).values()))
    record["settings"] = model.get_params(deep=False)
    return model


def validate_predictions(predictions, probabilities, task: str, classes=None) -> dict:
    import numpy as np

    predictions = np.asarray(predictions)
    if predictions.shape != (18,) or not np.isfinite(predictions).all():
        raise ValueError("Expected 18 finite predictions")
    details = {"prediction_shape": list(predictions.shape)}
    if task != "regression":
        classes = np.asarray(classes)
        expected_classes = np.arange(2 if task == "binary" else 3)
        probabilities = np.asarray(probabilities)
        if not np.array_equal(classes, expected_classes):
            raise ValueError("Unexpected probability column/class ordering")
        if not np.isin(predictions, classes).all():
            raise ValueError("Prediction contains an unknown class")
        if (probabilities.shape != (18, len(classes))
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("Invalid class probabilities")
        details["probability_shape"] = list(probabilities.shape)
        details["classes"] = classes.tolist()
    return details


def run_worker(args) -> dict:
    record = {"model": args.worker[0], "task": args.worker[1], "status": "failed"}
    started = time.monotonic()
    try:
        import numpy as np

        random.seed(SEED)
        np.random.seed(SEED)
        name, task = args.worker
        record["device"] = "cpu"
        if name in ("tabpfn", "mitra"):
            import torch

            torch.manual_seed(SEED)
            torch.set_num_threads(1)
            if args.device == "cuda" and not torch.cuda.is_available():
                raise Unavailable("CUDA was requested but is not available to PyTorch")
            record["device"] = (
                "cuda" if args.device == "auto" and torch.cuda.is_available()
                else "cpu" if args.device == "auto" else args.device
            )
        model = make_model(name, task, args, record)
        X_train, X_valid, X_test, y_train, y_valid = fixture(task)
        model.fit(X_train, y_train)
        if hasattr(model, "runtime_metadata_"):
            record["runtime"] = model.runtime_metadata_
            record["checkpoint"] = model.runtime_metadata_["checkpoint"]
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test) if task != "regression" else None
        # Mitra's sklearn interface does not expose classes_; fixture labels are contiguous.
        classes = getattr(model, "classes_", np.unique(y_train))
        record.update(validate_predictions(predictions, probabilities, task, classes))
        record["predictions"] = np.asarray(predictions).tolist()
        record["status"] = "passed"
    except (Unavailable, ImportError) as exc:
        record.update(status="unavailable", reason=str(exc))
    except Exception as exc:
        record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    record["seconds"] = round(time.monotonic() - started, 3)
    return record


def environment() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "torch",
                 "autogluon.tabular", "tabpfn", "huggingface-hub", "safetensors", "mitra-finetune"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "packages": packages}
    try:
        import torch

        info.update(cuda_available=torch.cuda.is_available(), torch_cuda_build=torch.version.cuda,
                    cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    cuda_memory_bytes=[torch.cuda.get_device_properties(i).total_memory
                                       for i in range(torch.cuda.device_count())])
    except ImportError:
        info["cuda_available"] = False
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--download", action="store_true", help="Download pinned checkpoints when missing")
    parser.add_argument("--timeout", type=int, default=180, help="Seconds allowed per model/task")
    parser.add_argument("--output", type=Path, default=ROOT / "results/runtime-smoke.json")
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    # Keep caches within the checkout and prevent implicit interactive authentication.
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface-home"))
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(ROOT / ".cache/tabpfn"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    if not args.download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        args.output.write_text(json.dumps(run_worker(args), indent=2) + "\n")
        return 0
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
              "purpose": "Synthetic runtime smoke test; not benchmark evaluation",
              "fixture": {"rows": 90, "features": 4, "train": 54, "validation": 18, "test": 18},
              "environment": environment(), "checks": []}
    for name in args.models:
        for task in args.tasks:
            print(f"Checking {name}/{task}...", flush=True)
            with tempfile.TemporaryDirectory(prefix="model-smoke-") as directory:
                output = Path(directory) / "result.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", name, task,
                           "--device", args.device, "--output", str(output)]
                if args.download:
                    command.append("--download")
                try:
                    process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                    if process.returncode != 0 or not output.exists():
                        check = dict(model=name, task=task, status="failed",
                                     reason=f"Worker exited with code {process.returncode} without a usable result")
                    else:
                        check = json.loads(output.read_text())
                except subprocess.TimeoutExpired:
                    check = dict(model=name, task=task, status="unavailable",
                                 reason=f"Exceeded {args.timeout}s runtime budget")
            report["checks"].append(check)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{name:10} {task:10} {check['status']}: {check.get('reason', 'prediction checks passed')}", flush=True)
    statuses = {check["status"] for check in report["checks"]}
    return 1 if "failed" in statuses else 2 if "unavailable" in statuses else 0


if __name__ == "__main__":
    raise SystemExit(main())
