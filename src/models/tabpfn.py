"""Frozen TabPFN-3.5-Fast adapters backed by the pinned checkpoint manifest."""

from importlib.metadata import version

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ..data import RANDOM_STATE, TASKS, Task
from .checkpoints import Unavailable, checkpoint


class _FrozenTabPFN(BaseEstimator):
    supports_tuning = False
    native_preprocessing = True

    def __init__(self, random_state=RANDOM_STATE, device="auto", n_estimators="auto",
                 download=False, fine_tune=False):
        self.random_state = random_state
        self.device = device
        self.n_estimators = n_estimators
        self.download = download
        self.fine_tune = fine_tune

    def fit(self, X, y):
        if self.fine_tune is not False:
            raise ValueError("Frozen TabPFN requires fine_tune=False")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("TabPFN device must be auto, cpu, or cuda")
        if self.n_estimators != "auto" and (
            not isinstance(self.n_estimators, int) or isinstance(self.n_estimators, bool)
            or self.n_estimators < 1
        ):
            raise ValueError("TabPFN n_estimators must be auto or a positive integer")
        try:
            import torch
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            from tabpfn.constants import ModelVersion
            from tabpfn.settings import settings
        except ImportError as exc:
            raise Unavailable("Install the declared torch and tabpfn dependencies before using TabPFN") from exc
        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        device = "cpu" if device == "auto" else device
        if device == "cuda" and not torch.cuda.is_available():
            raise Unavailable("TabPFN requested CUDA, but PyTorch cannot see a CUDA GPU; use device='cpu' for tiny fixtures")
        provenance = {}
        path = checkpoint("tabpfn", self.download, provenance)
        cls = TabPFNClassifier if isinstance(self, ClassifierMixin) else TabPFNRegressor
        self.model_ = cls.create_default_for_version(
            ModelVersion.V3_5_FAST, model_path=str(path), device=device,
            random_state=self.random_state, n_estimators=self.n_estimators,
            n_preprocessing_jobs=1, tuning_config=None, eval_metric=None,
            differentiable_input=False, ignore_pretraining_limits=False,
            fit_mode="fit_preprocessors",
        )
        try:
            # fit stores labeled context and fits preprocessing; no optimizer or
            # fine-tuning API is used. Suppress gradients throughout that path.
            with torch.no_grad():
                self.model_.fit(X, y)
        except torch.OutOfMemoryError as exc:
            raise Unavailable("TabPFN ran out of memory; no rows were silently dropped. Use more memory or a documented shared subsample.") from exc
        except RuntimeError as exc:
            if "Running on CPU with more than" in str(exc):
                raise Unavailable(f"TabPFN CPU configuration is unsupported: {exc}") from exc
            raise
        for module in self.model_.models_:
            module.requires_grad_(False)
            module.eval()
        self.n_features_in_ = self.model_.n_features_in_
        if hasattr(self.model_, "classes_"):
            self.classes_ = self.model_.classes_
        self.runtime_metadata_ = {
            **provenance, "model_version": "v3.5-fast", "device": device,
            "cuda_device_name": torch.cuda.get_device_name() if device == "cuda" else None,
            "n_estimators_resolved": self.model_.n_estimators_,
            "frozen_weights": True, "fine_tuning": False, "tuning_config": None,
            "differentiable_input": False, "ignore_pretraining_limits": False,
            "allow_cpu_large_dataset": settings.tabpfn.allow_cpu_large_dataset,
            "torch_threads": torch.get_num_threads(),
            "limits": {key: getattr(self.model_.inference_config_, key) for key in (
                "MAX_CPU_SAMPLES", "MAX_NUMBER_OF_SAMPLES", "MAX_NUMBER_OF_FEATURES", "MAX_NUMBER_OF_CLASSES")},
            "packages": {name: version(name) for name in ("tabpfn", "torch", "huggingface-hub", "safetensors")},
        }
        return self

    def predict(self, X):
        import torch

        check_is_fitted(self, "runtime_metadata_")
        try:
            with torch.no_grad():
                return self.model_.predict(X)
        except torch.OutOfMemoryError as exc:
            raise Unavailable("TabPFN prediction exceeded device memory; no rows were dropped or replacement scores generated") from exc


class FrozenTabPFNClassifier(ClassifierMixin, _FrozenTabPFN):
    def predict_proba(self, X):
        import torch

        check_is_fitted(self, "runtime_metadata_")
        try:
            with torch.no_grad():
                return self.model_.predict_proba(X)
        except torch.OutOfMemoryError as exc:
            raise Unavailable("TabPFN probability prediction exceeded device memory; no rows were dropped or replacement scores generated") from exc


class FrozenTabPFNRegressor(RegressorMixin, _FrozenTabPFN):
    pass


def make_tabpfn_models(task: Task, seed: int = RANDOM_STATE, device: str = "auto",
                       n_estimators: int | str = "auto", download: bool = False) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    cls = FrozenTabPFNRegressor if task == "regression" else FrozenTabPFNClassifier
    return {"TabPFN - Raw": cls(random_state=seed, device=device, n_estimators=n_estimators, download=download)}
