"""Checksum-pinned public dataset loading; downloaded archives stay out of Git."""

import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _read_pinned_member(slug: str, member_key: str, cache_dir, download: bool) -> tuple[dict, bytes]:
    """Verify both the downloaded archive and the selected source file before parsing."""
    metadata = json.loads((ROOT / f"datasets/{slug}.json").read_text())
    name = metadata["name"]
    cache = Path(cache_dir) if cache_dir is not None else ROOT / ".cache/datasets"
    archive = cache / f"{slug}.zip"
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f"{name} archive is not cached; enable dataset downloading")
        try:
            with urlopen(metadata["download_url"], timeout=60) as response:
                payload = response.read()
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Could not download the pinned {name} archive; check network access") from exc
        if hashlib.sha256(payload).hexdigest() != metadata["archive_sha256"]:
            raise ValueError(f"{name} download checksum does not match the pinned source")
        cache.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(payload)
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata["archive_sha256"]:
        raise ValueError(f"Cached {name} archive checksum mismatch; do not use this data")
    with ZipFile(BytesIO(payload)) as source:
        member = source.read(metadata[f"{member_key}_member"])
    if hashlib.sha256(member).hexdigest() != metadata[f"{member_key}_sha256"]:
        raise ValueError(f"{name} {member_key.upper()} checksum mismatch")
    return metadata, member


def load_online_shoppers(cache_dir: str | Path | None = None, download: bool = True) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load all released rows, excluding Revenue from X and preserving row order."""
    metadata, csv = _read_pinned_member("online-shoppers", "csv", cache_dir, download)
    frame = pd.read_csv(BytesIO(csv))
    target = metadata["target"]
    expected = metadata["numeric_features"] + metadata["categorical_features"] + [target]
    if len(frame) != metadata["rows"] or len(frame.columns) != len(expected) or set(frame) != set(expected):
        raise ValueError("Online Shoppers schema or row count does not match the pinned dataset")
    if frame.isna().any().any() or frame[target].dtype != bool or int(frame[target].sum()) != metadata["positive_rows"]:
        raise ValueError("Online Shoppers contains unexpected missing values or target labels/counts")
    numeric = frame[metadata["numeric_features"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("Online Shoppers numerical features must be finite and nonnegative")
    y = frame[target].copy()
    X = frame.drop(columns=target).copy()
    for column in metadata["categorical_features"]:
        # Codes identify categories, not ordered numerical magnitudes.
        X[column] = X[column].astype(str).astype("category")
    metadata["identical_full_rows"] = int(frame.duplicated().sum())
    metadata["excluded_columns"] = [target]
    metadata["sampling"] = "none; all source rows retained"
    return X, y, metadata


def load_seoul_bike(cache_dir: str | Path | None = None, download: bool = True) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load every hourly record; represent Date as days since the fixed Unix epoch.

    The date conversion is compatibility formatting, with no statistics learned from
    data. The target and source row order are preserved; non-functioning hours stay.
    """
    metadata, csv = _read_pinned_member("seoul-bike", "csv", cache_dir, download)
    frame = pd.read_csv(BytesIO(csv), encoding=metadata["encoding"])
    target = metadata["target"]
    expected = metadata["numeric_features"] + metadata["categorical_features"] + ["Date", target]
    if len(frame) != metadata["rows"] or len(frame.columns) != len(expected) or set(frame) != set(expected):
        raise ValueError("Seoul Bike schema or row count does not match the pinned dataset")
    if frame.isna().any().any():
        raise ValueError("Seoul Bike contains unexpected missing values")
    numeric = frame[metadata["numeric_features"] + [target]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (frame[target] < 0).any() or frame[target].dtype.kind not in "iu":
        raise ValueError("Seoul Bike needs finite predictors and nonnegative integer counts")
    dates = pd.to_datetime(frame["Date"], format=metadata["date_format"], errors="raise")
    timestamps = dates + pd.to_timedelta(frame["Hour"], unit="h")
    expected_times = pd.date_range(metadata["first_date"], periods=metadata["rows"], freq="h")
    if not frame["Hour"].between(0, 23).all() or not np.array_equal(timestamps, expected_times):
        raise ValueError("Seoul Bike must retain the complete ordered hourly source records")
    y = frame[target].copy()
    X = frame.drop(columns=target).copy()
    X["Date"] = (dates - pd.Timestamp("1970-01-01")).dt.days
    for column in metadata["categorical_features"]:
        X[column] = X[column].astype("category")
    metadata["date_representation"] = "integer days since 1970-01-01; parsed with %d/%m/%Y"
    metadata["identical_full_rows"] = int(frame.duplicated().sum())
    metadata["non_functioning_rows"] = int((frame["Functioning Day"] == "No").sum())
    metadata["excluded_columns"] = [target]
    metadata["sampling"] = "none; all source rows retained, including non-functioning hours"
    return X, y, metadata


def load_dry_bean(cache_dir: str | Path | None = None, download: bool = True) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Read the pinned ARFF release, preserving original columns, labels, and row order."""
    from scipy.io import arff

    metadata, payload = _read_pinned_member("dry-bean", "arff", cache_dir, download)
    records, _ = arff.loadarff(StringIO(payload.decode("utf-8")))
    frame = pd.DataFrame(records)
    target = metadata["target"]
    expected = metadata["numeric_features"] + [target]
    if len(frame) != metadata["rows"] or frame.columns.tolist() != expected:
        raise ValueError("Dry Bean schema, column order, or row count does not match the pinned dataset")
    frame[target] = frame[target].str.decode("utf-8")
    if frame.isna().any().any() or frame[target].value_counts().to_dict() != metadata["class_counts"]:
        raise ValueError("Dry Bean contains unexpected missing values or class labels/counts")
    X, y = frame.drop(columns=target).copy(), frame[target].copy()
    numeric = X.to_numpy(dtype=float)
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes) or not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError("Dry Bean predictors must be finite, strictly positive numerical measurements")
    metadata["identical_full_rows"] = int(frame.duplicated().sum())
    metadata["excluded_columns"] = [target]
    metadata["sampling"] = "none; all source rows retained, including identical records"
    return X, y, metadata
