"""Engineered features (issue #9).

Each feature is optional: a list of names selects which ones are added, so
that every feature can be evaluated on its own. Anything learned from data
(the median rate per grade) is learned in `fit`, i.e. on the training set only.

Features
--------
loan_to_income        loan_amnt / person_income (recomputed ratio; the provided
                      loan_percent_income differs from it on about 1% of rows).
rate_vs_grade_median  loan_int_rate minus the median rate of the same loan_grade:
                      a rate higher than usual for its grade signals a riskier file.
cred_hist_to_age      cb_person_cred_hist_length / person_age: share of adult life
                      with a credit history.
high_burden           1 if loan_percent_income > HIGH_BURDEN_THRESHOLD else 0
                      (threshold chosen from the training-set EDA).

The transformer expects loan_int_rate to be already imputed
(GroupMedianImputer runs before it in the preprocessing pipeline).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

ENGINEERED_FEATURES = (
    "loan_to_income",
    "rate_vs_grade_median",
    "cred_hist_to_age",
    "high_burden",
)
BINARY_ENGINEERED = ("high_burden",)
HIGH_BURDEN_THRESHOLD = 0.25

# Outcome of the cross-validated selection of issue #9 (see notebook 02, section 3).
# loan_to_income REPLACES the provided loan_percent_income: that column differs from the
# recomputed ratio on ~1% of rows, and ALL of those rows are defaults (a construction
# artifact of the dataset, i.e. target leakage). Using only the recomputed ratio removes it.
SELECTED_ENGINEERED: tuple = ("loan_to_income",)
SELECTED_DROP: tuple = ("loan_percent_income",)

_REQUIRED = {
    "loan_to_income": ["loan_amnt", "person_income"],
    "rate_vs_grade_median": ["loan_int_rate", "loan_grade"],
    "cred_hist_to_age": ["cb_person_cred_hist_length", "person_age"],
    "high_burden": ["loan_percent_income"],
}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Append the selected engineered features to a DataFrame."""

    def __init__(self, features=()):
        self.features = features

    def _check(self, X, features):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame")
        needed = {c for f in features for c in _REQUIRED[f]}
        missing = needed - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if "loan_to_income" in features and (X["person_income"] <= 0).any():
            raise ValueError("person_income must be strictly positive")
        if "cred_hist_to_age" in features and (X["person_age"] <= 0).any():
            raise ValueError("person_age must be strictly positive")
        if "rate_vs_grade_median" in features and X["loan_int_rate"].isna().any():
            raise ValueError("loan_int_rate must be imputed before computing rate_vs_grade_median")

    def fit(self, X, y=None):
        unknown = set(self.features) - set(ENGINEERED_FEATURES)
        if unknown:
            raise ValueError(f"Unknown engineered features: {sorted(unknown)}")
        self.features_ = tuple(f for f in ENGINEERED_FEATURES if f in set(self.features))
        self._check(X, self.features_)
        if "rate_vs_grade_median" in self.features_:
            self.grade_medians_ = X.groupby("loan_grade")["loan_int_rate"].median().to_dict()
            self.global_median_ = float(X["loan_int_rate"].median())
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "features_")
        self._check(X, self.features_)
        out = X.copy()
        if "loan_to_income" in self.features_:
            out["loan_to_income"] = out["loan_amnt"] / out["person_income"]
        if "rate_vs_grade_median" in self.features_:
            ref = out["loan_grade"].map(self.grade_medians_).astype(float).fillna(self.global_median_)
            out["rate_vs_grade_median"] = out["loan_int_rate"] - ref
        if "cred_hist_to_age" in self.features_:
            out["cred_hist_to_age"] = out["cb_person_cred_hist_length"] / out["person_age"]
        if "high_burden" in self.features_:
            out["high_burden"] = (out["loan_percent_income"] > HIGH_BURDEN_THRESHOLD).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "features_")
        return np.asarray(list(self.feature_names_in_) + list(self.features_), dtype=object)
