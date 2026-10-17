"""Small CPU tree baselines and fixed validation-search budgets."""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from ..data import RANDOM_STATE, TASKS, Task


class EncodedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Adapt XGBoost's contiguous-integer targets to the original class labels."""

    def __init__(self, n_estimators=80, max_depth=3, learning_rate=0.1,
                 random_state=RANDOM_STATE, n_jobs=1, tree_method="hist", device="cpu",
                 subsample=1.0, colsample_bytree=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.device = device
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def fit(self, X, y):
        from xgboost import XGBClassifier

        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.model_ = XGBClassifier(
            **self.get_params(),
            objective="binary:logistic" if len(self.classes_) == 2 else "multi:softprob",
        )
        self.model_.fit(X, self.encoder_.transform(y))
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.encoder_.inverse_transform(self.model_.predict(X).astype(int))

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        # Column i corresponds to encoder_.classes_[i]; no sorting at prediction time.
        return self.model_.predict_proba(X)


def make_classical_models(task: Task, seed: int = RANDOM_STATE) -> dict:
    """Construct unfitted models; all fitting remains inside the shared pipeline."""
    from lightgbm import LGBMClassifier, LGBMRegressor
    from xgboost import XGBRegressor

    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    common = dict(n_estimators=80, learning_rate=0.1, random_state=seed, n_jobs=1)
    xgb = (XGBRegressor(**common, max_depth=3, tree_method="hist", device="cpu",
                        objective="reg:squarederror", subsample=1.0, colsample_bytree=1.0)
           if task == "regression" else EncodedXGBClassifier(**common, max_depth=3))
    lgb_class = LGBMRegressor if task == "regression" else LGBMClassifier
    lgb = lgb_class(**common, num_leaves=15, min_child_samples=10, verbosity=-1,
                    deterministic=True, force_col_wise=True, device_type="cpu")
    return {"XGBoost": xgb, "LightGBM": lgb}


def classical_search_spaces() -> dict:
    """Four candidates per family, with 80 trees per candidate and no early stopping."""
    return {"XGBoost": {"learning_rate": [0.05, 0.1], "max_depth": [3, 6]},
            "LightGBM": {"learning_rate": [0.05, 0.1], "num_leaves": [7, 15]}}
