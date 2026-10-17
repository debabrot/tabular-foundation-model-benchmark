"""Default mixed-type preprocessing, fitted only by the training pipeline."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _categorical_values(frame: pd.DataFrame) -> pd.DataFrame:
    # Normalize None/pd.NA before SimpleImputer without learning from held-out rows.
    return frame.astype(object).where(frame.notna(), np.nan)


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Median-impute/scale numbers; mode-impute/one-hot categorical values."""
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    if set(numeric + categorical) != set(X_train.columns):
        raise ValueError("Convert unsupported column types (such as dates) into explicit features first")
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("normalize", FunctionTransformer(_categorical_values)),
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop")
