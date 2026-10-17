"""Deterministic positional splits and dataset fingerprints."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Task = Literal["binary", "multiclass", "regression"]
RANDOM_STATE = 42
TASKS = ("binary", "multiclass", "regression")


def validate_split(splits: dict, y: np.ndarray, task: Task) -> None:
    """Require exhaustive, disjoint row positions and usable evaluation partitions."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Splits must contain train, validation, and test")
    arrays = [np.asarray(splits[key]) for key in ("train", "validation", "test")]
    if any(a.ndim != 1 or len(a) < 2 or a.dtype.kind not in "iu" for a in arrays):
        raise ValueError("Each split needs at least two integer row positions")
    combined = np.concatenate(arrays)
    if not np.array_equal(np.sort(combined), np.arange(len(y))):
        raise ValueError("Split positions must cover every row exactly once")
    if task != "regression":
        classes = np.unique(y)
        if (task == "binary" and len(classes) != 2) or (task == "multiclass" and len(classes) < 3):
            raise ValueError("Target class count does not match the task")
        if any(not np.array_equal(np.unique(y[a]), classes) for a in arrays):
            raise ValueError("Every partition must contain all target classes")


def make_split(y: np.ndarray, task: Task, seed: int = RANDOM_STATE) -> dict[str, np.ndarray]:
    """Create one 60/20/20 split, stratified for classification."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    y = np.asarray(y)
    positions = np.arange(len(y))
    train, holdout = train_test_split(
        positions, test_size=0.4, random_state=seed,
        stratify=y if task != "regression" else None,
    )
    validation, test = train_test_split(
        holdout, test_size=0.5, random_state=seed,
        stratify=y[holdout] if task != "regression" else None,
    )
    splits = {"train": train, "validation": validation, "test": test}
    validate_split(splits, y, task)
    return splits


def dataset_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    """Hash ordered values, indices, columns, and dtypes; not a raw-download checksum."""
    digest = hashlib.sha256()
    schema = {"columns": X.columns.tolist(), "dtypes": list(map(str, X.dtypes)),
              "target_dtype": str(y.dtype), "target_name": y.name}
    digest.update(json.dumps(schema, sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    digest.update(pd.util.hash_pandas_object(y, index=True).to_numpy().tobytes())
    return digest.hexdigest()
