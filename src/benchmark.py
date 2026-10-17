"""Run selected estimators on a common split and persist auditable results."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import time
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import FunctionTransformer

from .data import RANDOM_STATE, Task, dataset_fingerprint, make_split, validate_split
from .evaluation import PRIMARY_METRICS, evaluate_predictions
from .preprocessing import make_preprocessor


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}; use explicit JSON-compatible model settings")


def _configuration(value):
    """Describe estimator settings without memory addresses or non-JSON defaults."""
    if isinstance(value, BaseEstimator):
        return {"estimator": f"{type(value).__module__}.{type(value).__name__}",
                "parameters": _configuration(value.get_params(deep=False))}
    if isinstance(value, dict):
        return {key: _configuration(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_configuration(item) for item in value]
    if isinstance(value, np.generic):
        return _configuration(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"float": str(value)}
    if callable(value):
        return {"callable": f"{value.__module__}.{value.__qualname__}"}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported setting {type(value).__name__}; use explicit reproducible parameters")


def _evaluate_partition(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray,
                        task: Task, positive_label) -> tuple[dict, dict]:
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X) if task != "regression" else None
    metrics = evaluate_predictions(
        task, y, predictions, probabilities=probabilities,
        classes=getattr(pipeline.named_steps["model"], "classes_", None),
        positive_label=positive_label,
    )
    return metrics, {"values": predictions, "probabilities": probabilities}


def run_comparison(
    X: pd.DataFrame, y: pd.Series | np.ndarray, models: Mapping[str, BaseEstimator], *,
    task: Task, dataset_name: str, seed: int = RANDOM_STATE, splits: dict | None = None,
    preprocessor: BaseEstimator | None = None, positive_label=None,
    dataset_kind: str = "dataset",
    search_spaces: Mapping[str, dict] | None = None,
    preprocessors: Mapping[str, BaseEstimator] | None = None,
) -> dict:
    """Fit on train, optionally select by validation, then evaluate final test.

    Search spaces define finite parameter grids. Every candidate fits preprocessing
    on train only. All model selections finish before any test predictions are made.
    Pass saved positional splits for later comparisons. Caller estimators are cloned.
    """
    if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
        raise ValueError("X must be a nonempty DataFrame with unique columns")
    if not all(isinstance(column, str) for column in X.columns):
        raise ValueError("Feature names must be strings")
    if isinstance(y, pd.Series) and not y.index.equals(X.index):
        raise ValueError("Target and feature indices must align")
    y = pd.Series(np.asarray(y), index=X.index, name=getattr(y, "name", None))
    if len(y) != len(X) or y.isna().any() or not models:
        raise ValueError("Provide aligned, nonmissing targets and at least one model")
    search_spaces = {} if search_spaces is None else search_spaces
    if set(search_spaces) - set(models):
        raise ValueError("Search spaces must name models included in the comparison")
    preprocessors = {} if preprocessors is None else preprocessors
    if set(preprocessors) - set(models):
        raise ValueError("Preprocessing overrides must name models included in the comparison")
    for name in search_spaces:
        if not getattr(models[name], "supports_tuning", True):
            raise ValueError(f"{name} is a frozen inference model; validation tuning is disabled")
    split_method = "provided" if splits is not None else "random-stratified" if task != "regression" else "random"
    splits = make_split(y.to_numpy(), task, seed) if splits is None else splits
    validate_split(splits, y.to_numpy(), task)
    splits = {key: np.asarray(value) for key, value in splits.items()}
    train = splits["train"]
    if task == "regression" and not np.isfinite(y.to_numpy()).all():
        raise ValueError("Regression targets must be finite")
    if task == "binary":
        classes = np.unique(y.iloc[train])
        positive_label = classes[-1] if positive_label is None else positive_label
        if positive_label not in classes:
            raise ValueError("positive_label must be a training class")
    else:
        positive_label = None
    use_default_preprocessing = preprocessor is None
    if use_default_preprocessing:
        needs_default = any(name not in preprocessors and not getattr(model, "native_preprocessing", False)
                            for name, model in models.items())
        preprocessor = make_preprocessor(X.iloc[train]) if needs_default else FunctionTransformer()
    results = {
        "schema_version": 1, "dataset": {"name": dataset_name, "kind": dataset_kind,
                                           "fingerprint_sha256": dataset_fingerprint(X, y)},
        "task": task, "seed": seed, "split_method": split_method, "splits": splits,
        "primary_metric": PRIMARY_METRICS[task], "positive_label": positive_label,
        "preprocessing": _configuration(preprocessor),
        "environment": {"python": platform.python_version(), "packages": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")}},
        "targets": {key: y.iloc[splits[key]].to_numpy() for key in ("validation", "test")},
        "models": [],
    }
    for package in ("xgboost", "lightgbm"):
        try:
            results["environment"]["packages"][package] = version(package)
        except PackageNotFoundError:
            pass
    selected_pipelines = []
    for name, estimator in models.items():
        model_preprocessor = preprocessors.get(name, preprocessor)
        if name not in preprocessors and use_default_preprocessing and getattr(estimator, "native_preprocessing", False):
            model_preprocessor = FunctionTransformer()
        candidates = list(ParameterGrid(search_spaces[name])) if name in search_spaces else [{}]
        best_score, selected, trials = None, None, []
        for candidate in candidates:
            model = clone(estimator).set_params(**candidate)
            if "random_state" in model.get_params(deep=False) and model.get_params()["random_state"] is None:
                model.set_params(random_state=seed)
            pipeline = Pipeline([("preprocessing", clone(model_preprocessor)), ("model", model)])
            started = time.monotonic()
            pipeline.fit(X.iloc[train], y.iloc[train])
            validation_metrics, validation_predictions = _evaluate_partition(
                pipeline, X.iloc[splits["validation"]], results["targets"]["validation"], task, positive_label,
            )
            score = validation_metrics[PRIMARY_METRICS[task]]
            trials.append({"parameters": candidate, "validation_metrics": validation_metrics,
                           "fit_and_validation_seconds": time.monotonic() - started})
            # Strict improvement makes ties deterministic: first candidate wins.
            if best_score is None or (score < best_score if task == "regression" else score > best_score):
                best_score = score
                selected = (pipeline, validation_metrics, validation_predictions, len(trials) - 1)
        if selected is None:
            raise ValueError(f"Search for {name} produced no candidates")
        pipeline, validation_metrics, validation_predictions, selected_index = selected
        model = pipeline.named_steps["model"]
        row = {"name": name, "estimator": f"{type(model).__module__}.{type(model).__name__}",
               "parameters": _configuration(model.get_params(deep=False)),
               "metrics": {"validation": validation_metrics},
               "predictions": {"validation": validation_predictions},
               "classes": getattr(model, "classes_", None)}
        if isinstance(getattr(model, "model_", None), BaseEstimator):
            row["underlying_model"] = _configuration(model.model_)
        if model_preprocessor is not preprocessor:
            row["preprocessing"] = _configuration(model_preprocessor)
        if hasattr(model, "runtime_metadata_"):
            row["runtime"] = _configuration(model.runtime_metadata_)
        if name in search_spaces:
            row["tuning"] = {"method": "grid-validation", "search_space": search_spaces[name],
                             "metric": PRIMARY_METRICS[task],
                             "direction": "minimize" if task == "regression" else "maximize",
                             "candidate_count": len(trials), "selected_index": selected_index,
                             "tie_break": "first candidate", "trials": trials}
        results["models"].append(row)
        selected_pipelines.append(pipeline)
    for row, pipeline in zip(results["models"], selected_pipelines):
        row["metrics"]["test"], row["predictions"]["test"] = _evaluate_partition(
            pipeline, X.iloc[splits["test"]], results["targets"]["test"], task, positive_label,
        )
    # Return the same types callers receive after loading the JSON artifact.
    return json.loads(json.dumps(results, default=_json_default, allow_nan=False))


def save_results(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_results(path: str | Path) -> dict:
    results = json.loads(Path(path).read_text())
    if results.get("schema_version") != 1:
        raise ValueError("Unsupported result schema")
    return results
