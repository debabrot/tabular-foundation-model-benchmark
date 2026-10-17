"""Label-free, row-local feature construction for dataset experiments."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ShopperFeatures(TransformerMixin, BaseEstimator):
    """Keep original columns and add five fixed session-level features."""

    required = ("Administrative", "Informational", "ProductRelated",
                "Administrative_Duration", "Informational_Duration", "ProductRelated_Duration",
                "ExitRates", "BounceRates")
    added = ("TotalPages", "TotalDuration", "SecondsPerPage", "ProductPageFraction", "ExitBounceGap")

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame) or not set(self.required).issubset(X.columns):
            raise ValueError("ShopperFeatures requires the named session page, duration, and rate columns")
        if set(self.added) & set(X.columns):
            raise ValueError("Engineered shopper feature names already exist")
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        if not np.array_equal(X.columns, self.feature_names_in_):
            raise ValueError("Shopper feature columns must match fit order")
        result = X.copy(deep=True)
        result["TotalPages"] = X[["Administrative", "Informational", "ProductRelated"]].sum(axis=1)
        result["TotalDuration"] = X[["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]].sum(axis=1)
        denominator = result["TotalPages"].replace(0, np.nan)
        result["SecondsPerPage"] = (result["TotalDuration"] / denominator).fillna(0)
        result["ProductPageFraction"] = (X["ProductRelated"] / denominator).fillna(0)
        result["ExitBounceGap"] = X["ExitRates"] - X["BounceRates"]
        return result


class BikeFeatures(TransformerMixin, BaseEstimator):
    """Keep original predictors and add fixed, row-local calendar features.

    Date is the loader's integer Unix-day representation. No rental counts, lags,
    aggregates, or validation/test statistics are used.
    """

    added = ("HourSin", "HourCos", "DayOfWeek", "Month", "IsWeekend", "DayOfYearSin", "DayOfYearCos")

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame) or not {"Date", "Hour"}.issubset(X.columns):
            raise ValueError("BikeFeatures requires Date (Unix days) and Hour columns")
        if set(self.added) & set(X.columns):
            raise ValueError("Engineered bike feature names already exist")
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        if not np.array_equal(X.columns, self.feature_names_in_):
            raise ValueError("Bike feature columns must match fit order")
        dates = pd.to_datetime(X["Date"], unit="D", origin="unix", errors="raise")
        result = X.copy(deep=True)
        result["HourSin"] = np.sin(2 * np.pi * X["Hour"] / 24)
        result["HourCos"] = np.cos(2 * np.pi * X["Hour"] / 24)
        result["DayOfWeek"] = dates.dt.dayofweek
        result["Month"] = dates.dt.month
        result["IsWeekend"] = dates.dt.dayofweek >= 5
        result["DayOfYearSin"] = np.sin(2 * np.pi * (dates.dt.dayofyear - 1) / 365.25)
        result["DayOfYearCos"] = np.cos(2 * np.pi * (dates.dt.dayofyear - 1) / 365.25)
        return result


class BeanFeatures(TransformerMixin, BaseEstimator):
    """Keep source predictors and add four fixed log-size representations."""

    required = ("Area", "Perimeter", "MajorAxisLength", "MinorAxisLength")
    added = ("LogArea", "LogPerimeter", "LogMajorAxisLength", "LogMinorAxisLength")

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame) or not set(self.required).issubset(X.columns):
            raise ValueError("BeanFeatures requires the source area, perimeter, and axis-length columns")
        if set(self.added) & set(X.columns):
            raise ValueError("Engineered bean feature names already exist")
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        if not np.array_equal(X.columns, self.feature_names_in_):
            raise ValueError("Bean feature columns must match fit order")
        values = X[list(self.required)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Bean log-size features require finite, strictly positive measurements")
        result = X.copy(deep=True)
        result[list(self.added)] = np.log(values)
        return result
