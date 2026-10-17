"""Forward-only Mitra v2 inference, including the distributional regression head."""

from importlib.metadata import version
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from ..data import RANDOM_STATE, TASKS, Task
from ..preprocessing import _categorical_values
from .checkpoints import Unavailable, checkpoint


def _native_preprocessor(regression: bool, n_classes: int, seed: int):
    from autogluon.tabular.models.mitra._internal.config.enums import Task as MitraTask
    from autogluon.tabular.models.mitra._internal.data.preprocessor import Preprocessor

    rng = np.random.RandomState(seed)

    class SeededPreprocessor(Preprocessor):
        # The upstream defaults use NumPy's global RNG for these two operations.
        # Keep the same transformations with a local RNG and no global-state edits.
        def determine_mirror(self, X):
            self.mirror = rng.choice([1, -1], size=(1, X.shape[1]))

        def determine_regression_mirror(self):
            self.regression_mirror = bool(rng.choice([True, False]))

    return SeededPreprocessor(
        dim_embedding=None, n_classes=n_classes, dim_output=1000 if regression else 10,
        use_quantile_transformer=False, use_feature_count_scaling=False,
        use_random_transforms=False, shuffle_classes=False, shuffle_features=False,
        random_mirror_regression=True, random_mirror_x=True,
        task=MitraTask.REGRESSION if regression else MitraTask.CLASSIFICATION,
    )


def regression_mean(logits, edges):
    """Decode a distributional head by its mean, not its most likely bin."""
    import torch

    centers = (edges[:-1] + edges[1:]) / 2
    return (torch.softmax(logits.float(), dim=-1) * centers).sum(dim=-1)


class _FrozenMitra(BaseEstimator):
    supports_tuning = False
    native_preprocessing = True

    def __init__(self, random_state=RANDOM_STATE, device="auto", download=False,
                 fine_tune=False, batch_size=256, max_support_rows=8192, max_features=500):
        self.random_state = random_state
        self.device = device
        self.download = download
        self.fine_tune = fine_tune
        self.batch_size = batch_size
        self.max_support_rows = max_support_rows
        self.max_features = max_features

    def fit(self, X, y):
        if self.fine_tune is not False:
            raise ValueError("Frozen Mitra requires fine_tune=False")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Mitra device must be auto, cpu, or cuda")
        for name in ("batch_size", "max_support_rows", "max_features"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Mitra {name} must be a positive integer")
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise ValueError("Mitra random_state must be an integer")
        frame = X.copy(deep=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(np.array(X, copy=True))
        y = np.asarray(y)
        if frame.empty or not frame.columns.is_unique or y.shape != (len(frame),) or pd.isna(y).any():
            raise ValueError("Mitra needs nonempty features with unique columns and aligned, nonmissing targets")
        if len(frame) > self.max_support_rows or frame.shape[1] > self.max_features:
            raise Unavailable("Mitra input exceeds the configured support-row or feature limit; no rows or features were sampled")
        regression = isinstance(self, RegressorMixin)
        if regression:
            y = y.astype(np.float32)
            if not np.isfinite(y).all() or y.min() == y.max():
                raise ValueError("Mitra regression requires finite, varying training targets")
            n_classes = 0
        else:
            self.label_encoder_ = LabelEncoder().fit(y)
            self.classes_ = self.label_encoder_.classes_
            n_classes = len(self.classes_)
            if not 2 <= n_classes <= 10:
                raise Unavailable("The pinned Mitra v2 classifier supports 2–10 classes; no class decomposition is enabled")
            y = self.label_encoder_.transform(y)
        try:
            import torch
            from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D
        except ImportError as exc:
            raise Unavailable("Install the declared autogluon.tabular[mitra], torch, and safetensors dependencies") from exc
        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        self.device_ = "cpu" if device == "auto" else device
        if self.device_ == "cuda" and not torch.cuda.is_available():
            raise Unavailable("Mitra requested CUDA, but PyTorch cannot see a CUDA GPU")
        # Match the native trainer's GPU autocast policy while retaining stable
        # float32 CPU inference. Parameters remain float32 and are never updated.
        self.autocast_dtype_ = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.device_ == "cuda" else None
        provenance = {}
        path = checkpoint("mitra-regression" if regression else "mitra-classification", self.download, provenance)
        config = json.loads((path / "config.json").read_text())
        expected_task, expected_head = ("REGRESSION", 1000) if regression else ("CLASSIFICATION", 10)
        if config.get("task") != expected_task or config.get("dim_output") != expected_head:
            raise Unavailable("Mitra checkpoint configuration does not match the required v2 task/head")
        numeric = frame.select_dtypes(include="number").columns.tolist()
        categorical = frame.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
        if set(numeric + categorical) != set(frame.columns):
            raise ValueError("Convert unsupported Mitra column types, such as dates, into explicit features first")
        self.input_encoder_ = ColumnTransformer([
            ("numeric", "passthrough", numeric),
            ("categorical", Pipeline([
                ("normalize", FunctionTransformer(_categorical_values)),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                            encoded_missing_value=np.nan)),
            ]), categorical),
        ])
        values = np.asarray(self.input_encoder_.fit_transform(frame), dtype=np.float32)
        if np.isinf(values).any():
            raise ValueError("Mitra numeric features must not contain infinity")
        self.preprocessor_ = _native_preprocessor(regression, n_classes, self.random_state)
        self.preprocessor_.fit(values.copy(), y.copy())
        self.support_X_ = self.preprocessor_.transform_X(values.copy())
        self.support_y_ = self.preprocessor_.transform_y(y.copy())
        if self.support_X_.shape[1] == 0:
            raise Unavailable("Mitra has no varying training features after native preprocessing")
        try:
            # Loading initializes a temporary CPU network before replacing all weights.
            # Preserve the caller's RNG state; no trainer/optimizer is constructed.
            with torch.random.fork_rng(devices=[]):
                self.model_ = Tab2D.from_pretrained(str(path), device=self.device_).to(self.device_)
            self.model_.requires_grad_(False)
            self.model_.eval()
        except torch.OutOfMemoryError as exc:
            raise Unavailable("Mitra weights do not fit in device memory; no fallback model was used") from exc
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.array(frame.columns)
        self.regression_ = regression
        self.edges_ = torch.linspace(-0.5, 1.5, 1001, device=self.device_) if regression else None
        self.runtime_metadata_ = {
            **provenance, "model_version": "mitra-v2", "architecture": config,
            "device": self.device_, "cuda_device_name": torch.cuda.get_device_name() if self.device_ == "cuda" else None,
            "frozen_weights": True, "fine_tuning": False, "optimizer_created": False,
            "n_estimators_resolved": 1,
            "precision": "bfloat16-autocast" if self.autocast_dtype_ == torch.bfloat16 else "float16-autocast" if self.autocast_dtype_ else "float32",
            "torch_threads": torch.get_num_threads(),
            "support_rows": len(frame), "support_sampling": "none", "query_batch_size": self.batch_size,
            "max_support_rows": self.max_support_rows, "max_features": self.max_features,
            "feature_encoding": "train-fitted ordinal categories; unknown/missing become NaN",
            "numeric_columns": numeric, "categorical_columns": categorical,
            "imputation_means": self.preprocessor_.pre_nan_mean.tolist(),
            "singular_features": self.preprocessor_.singular_features.tolist(),
            "feature_mirror": self.preprocessor_.mirror.tolist(),
            "regression_decode": "softmax-weighted bin mean" if regression else None,
            "target_min": float(y.min()) if regression else None,
            "target_max": float(y.max()) if regression else None,
            "target_mirror": self.preprocessor_.regression_mirror if regression else None,
            "packages": {name: version(name) for name in ("autogluon.tabular", "torch", "huggingface-hub", "safetensors")},
        }
        return self

    def _logits(self, X):
        import torch

        check_is_fitted(self, "runtime_metadata_")
        frame = X.copy(deep=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(np.array(X, copy=True))
        if not np.array_equal(frame.columns, self.feature_names_in_):
            raise ValueError("Mitra prediction columns must match the training columns in order")
        values = np.asarray(self.input_encoder_.transform(frame), dtype=np.float32)
        if np.isinf(values).any():
            raise ValueError("Mitra numeric features must not contain infinity")
        query = self.preprocessor_.transform_X(values.copy())
        support = torch.tensor(self.support_X_, device=self.device_).unsqueeze(0)
        targets = torch.tensor(self.support_y_, device=self.device_).unsqueeze(0)
        if self.regression_:
            targets = (torch.bucketize(targets, self.edges_) - 1).clamp(0, 999).long()
        output = []
        try:
            with torch.no_grad(), torch.autocast(
                device_type=self.device_, dtype=self.autocast_dtype_ or torch.bfloat16,
                enabled=self.autocast_dtype_ is not None,
            ):
                for start in range(0, len(query), self.batch_size):
                    batch = torch.tensor(query[start:start + self.batch_size], device=self.device_).unsqueeze(0)
                    logits = self.model_(
                        support.clone(), targets.clone(), batch,
                        torch.zeros((1, support.shape[2]), dtype=torch.bool, device=self.device_),
                        torch.zeros((1, support.shape[1]), dtype=torch.bool, device=self.device_),
                        torch.zeros((1, batch.shape[1]), dtype=torch.bool, device=self.device_),
                    )[0].float()
                    if not torch.isfinite(logits).all():
                        raise ValueError("Mitra produced nonfinite logits; no replacement scores were generated")
                    output.append(logits.cpu())
        except torch.OutOfMemoryError as exc:
            raise Unavailable("Mitra inference exceeded device memory; reduce query batch size or use more memory; context rows were not dropped") from exc
        if not output:
            raise ValueError("Mitra requires at least one query row")
        return torch.cat(output)

    def predict(self, X):
        check_is_fitted(self, "runtime_metadata_")
        if isinstance(self, ClassifierMixin):
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        logits = self._logits(X)
        values = regression_mean(logits, self.edges_.cpu()).numpy()
        return self.preprocessor_.inverse_transform_y(values)


class FrozenMitraClassifier(ClassifierMixin, _FrozenMitra):
    def predict_proba(self, X):
        import torch

        logits = self._logits(X)[:, :len(self.classes_)]
        return torch.softmax(logits, dim=-1).numpy()


class FrozenMitraRegressor(RegressorMixin, _FrozenMitra):
    pass


def make_mitra_models(task: Task, seed: int = RANDOM_STATE, device: str = "auto",
                      download: bool = False, batch_size: int = 256) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    cls = FrozenMitraRegressor if task == "regression" else FrozenMitraClassifier
    return {"Mitra v2 - Raw": cls(random_state=seed, device=device, download=download, batch_size=batch_size)}
