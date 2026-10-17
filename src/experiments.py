"""Bound model runs in separate processes while preserving common dataset splits."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
import multiprocessing
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .benchmark import _configuration, run_comparison, save_results
from .data import RANDOM_STATE, dataset_fingerprint, make_split, validate_split
from .evaluation import PRIMARY_METRICS
from .models.checkpoints import Unavailable


def _model_worker(path: str, X: pd.DataFrame, y: pd.Series, name: str, model: BaseEstimator, kwargs: dict):
    try:
        import torch

        torch.set_num_threads(1)
        result = run_comparison(X, y, {name: model}, **kwargs)
        payload = {"status": "evaluated", "results": result}
    except Unavailable as exc:
        payload = {"status": "unavailable", "reason": str(exc)}
    except Exception as exc:
        payload = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    Path(path).write_text(json.dumps(payload, allow_nan=False))


def run_isolated_comparison(
    X: pd.DataFrame, y: pd.Series, models: dict[str, BaseEstimator], *,
    task: str, dataset_name: str, splits: dict | None = None, seed: int = RANDOM_STATE,
    positive_label=None, search_spaces: dict | None = None, preprocessors: dict | None = None,
    timeout: float = 300, output_path: str | Path | None = None,
) -> dict:
    """Use predeclared independent recipes; score each winner once on its test split.

    A timeout/unavailable model gets no metrics. Unexpected failures are distinguished
    from unavailability. Each process exits after its model so GPU memory is released.
    """
    if not models or timeout <= 0:
        raise ValueError("Provide models and a positive per-model timeout")
    if not isinstance(y, pd.Series) or not X.index.equals(y.index) or y.isna().any():
        raise ValueError("Targets must be a nonmissing Series aligned with the feature index")
    splits = make_split(y.to_numpy(), task, seed) if splits is None else splits
    validate_split(splits, y.to_numpy(), task)
    search_spaces, preprocessors = search_spaces or {}, preprocessors or {}
    if (set(search_spaces) | set(preprocessors)) - set(models):
        raise ValueError("Search/preprocessing overrides must name included models")
    if task == "binary":
        classes = np.unique(y.iloc[splits["train"]])
        positive_label = classes[-1] if positive_label is None else positive_label
        if positive_label not in classes:
            raise ValueError("positive_label must be a training class")
    report = {
        "schema_version": 1, "dataset": {"name": dataset_name, "kind": "dataset",
            "fingerprint_sha256": dataset_fingerprint(X, y)},
        "task": task, "seed": seed, "split_method": "provided", "splits": _configuration(splits),
        "positive_label": _configuration(positive_label), "primary_metric": PRIMARY_METRICS[task],
        "targets": {partition: y.iloc[splits[partition]].tolist() for partition in ("validation", "test")},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": {}},
        "execution": {"isolation": "one spawned process per model", "timeout_seconds": timeout,
                      "test_policy": "independent predeclared recipes; no test-based selection across models"},
        "models": [],
    }
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "xgboost", "lightgbm", "torch", "tabpfn", "autogluon.tabular"):
        try:
            report["environment"]["packages"][package] = version(package)
        except PackageNotFoundError:
            pass
    context = multiprocessing.get_context("spawn")
    for name, model in models.items():
        print(f"Running {name}...", flush=True)
        kwargs = dict(task=task, dataset_name=dataset_name, seed=seed, splits=splits, positive_label=positive_label,
                      search_spaces={name: search_spaces[name]} if name in search_spaces else None,
                      preprocessors={name: preprocessors[name]} if name in preprocessors else None)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="benchmark-model-") as directory:
            result_path = Path(directory) / "result.json"
            process = context.Process(target=_model_worker, args=(str(result_path), X, y, name, model, kwargs))
            process.start()
            try:
                process.join(timeout)
                if process.is_alive():
                    payload = {"status": "unavailable", "reason": f"Exceeded the {timeout:g}s per-model runtime budget"}
                elif not result_path.exists():
                    payload = {"status": "failed", "reason": f"Worker exited with code {process.exitcode} without a result"}
                else:
                    payload = json.loads(result_path.read_text())
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                    if process.is_alive():
                        process.kill()
                        process.join()
                process.close()
        if payload["status"] == "evaluated":
            result = payload["results"]
            if result["splits"] != report["splits"] or result["dataset"]["fingerprint_sha256"] != report["dataset"]["fingerprint_sha256"]:
                raise ValueError("A worker changed the shared dataset or split")
            row = result["models"][0]
            row.setdefault("preprocessing", result["preprocessing"])
            row["status"] = "evaluated"
        else:
            row = {"name": name, "status": payload["status"], "reason": payload["reason"],
                   "parameters": _configuration(model.get_params(deep=False)), "metrics": {}, "predictions": {}}
        row["elapsed_seconds"] = time.monotonic() - started
        report["models"].append(row)
        if output_path is not None:
            save_results(report, output_path)
        print(f"{name}: {row['status']}" + (f" — {row['reason']}" if "reason" in row else ""), flush=True)
    return report
