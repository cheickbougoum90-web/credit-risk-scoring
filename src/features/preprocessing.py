"""Preprocessing pipeline (issue #7).

Implements the validated cleaning decisions that must be LEARNED on the
training set (so they cannot live in src/data/clean.py):

* loan_int_rate: median imputation by loan_grade (fallback: global median).
* person_emp_length: median imputation + a missing-value indicator.
* person_income: log-transform and scaling for linear models only.
* loan_grade and cb_person_default_on_file: ordinal encoding.
* other categoricals: one-hot encoding.

Usage:
    pipe = build_preprocessor("tree")      # or "linear"
    X_train_t = pipe.fit_transform(X_train)
    X_val_t = pipe.transform(X_val)        # nothing is re-learned here
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

from src.data.load import TARGET
from src.features.engineering import (
    BINARY_ENGINEERED,
    ENGINEERED_FEATURES,
    SELECTED_DROP,
    SELECTED_ENGINEERED,
    FeatureEngineer,
)

NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
ORDINAL_COLS = ["loan_grade", "cb_person_default_on_file"]
NOMINAL_COLS = ["person_home_ownership", "loan_intent"]
FEATURES = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS

GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
DEFAULT_FLAG_ORDER = ["N", "Y"]
MODEL_FAMILIES = ("tree", "linear")


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model input columns, in a fixed order."""
    return df[FEATURES].copy()


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labelled DataFrame into features and target."""
    return get_features(df), df[TARGET].copy()


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill missing values of `column` with the median of its `group_column`.

    Medians are learned in `fit` (training set only). Groups unseen in `fit`,
    or whose values were all missing, fall back to the global median.
    """

    def __init__(self, column: str = "loan_int_rate", group_column: str = "loan_grade"):
        self.column = column
        self.group_column = group_column

    def _check_frame(self, X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("GroupMedianImputer expects a pandas DataFrame")
        missing = {self.column, self.group_column} - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def fit(self, X, y=None):
        self._check_frame(X)
        values = X[self.column]
        if values.notna().sum() == 0:
            raise ValueError(f"Column {self.column!r} has no observed value to learn from")
        self.global_median_ = float(values.median())
        self.group_medians_ = values.groupby(X[self.group_column]).median().dropna().to_dict()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "group_medians_")
        self._check_frame(X)
        out = X.copy()
        fill = out[self.group_column].map(self.group_medians_).astype(float)
        fill = fill.fillna(self.global_median_)
        out[self.column] = out[self.column].fillna(fill)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_


# Module-level functions (not lambdas) so the fitted pipeline can be pickled.
def _missing_flag(X):
    return pd.DataFrame(X).isna().astype(float)


def _missing_flag_names(transformer, input_features):
    return [f"{name}_missing" for name in input_features]


def build_preprocessor(model_family: str = "tree", engineered=(), drop=()) -> Pipeline:
    """Build the (unfitted) preprocessing pipeline.

    model_family="tree": raw numeric scales (trees are scale-invariant).
    model_family="linear": log1p on income and standard scaling of numerics.
    engineered: names of engineered features to add (see src/features/engineering.py).
    drop: generic numeric columns to remove (e.g. ["loan_percent_income"]).
    """
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"model_family must be one of {MODEL_FAMILIES}, got {model_family!r}")
    scale = model_family == "linear"

    income_tf = (
        Pipeline(
            [
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        if scale
        else "passthrough"
    )

    emp_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        emp_steps.append(("scale", StandardScaler()))

    generic_numeric = [c for c in NUMERIC_COLS if c not in ("person_income", "person_emp_length")]
    bad_drop = set(drop) - set(generic_numeric)
    if bad_drop:
        raise ValueError(f"Cannot drop {sorted(bad_drop)}; allowed: {generic_numeric}")
    unknown = set(engineered) - set(ENGINEERED_FEATURES)
    if unknown:
        raise ValueError(f"Unknown engineered features: {sorted(unknown)}")
    engineered = tuple(f for f in ENGINEERED_FEATURES if f in set(engineered))
    other_numeric = [c for c in generic_numeric if c not in drop] + [
        f for f in engineered if f not in BINARY_ENGINEERED
    ]
    binary_engineered = [f for f in engineered if f in BINARY_ENGINEERED]

    columns = ColumnTransformer(
        transformers=[
            ("income", income_tf, ["person_income"]),
            ("emp_length", Pipeline(emp_steps), ["person_emp_length"]),
            (
                "emp_length_missing",
                FunctionTransformer(_missing_flag, feature_names_out=_missing_flag_names),
                ["person_emp_length"],
            ),
            ("numeric", StandardScaler() if scale else "passthrough", other_numeric),
            *([("engineered_binary", "passthrough", binary_engineered)] if binary_engineered else []),
            (
                "ordinal",
                OrdinalEncoder(categories=[GRADE_ORDER, DEFAULT_FLAG_ORDER]),
                ORDINAL_COLS,
            ),
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                NOMINAL_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(
        [
            ("rate_imputer", GroupMedianImputer()),
            ("engineer", FeatureEngineer(features=engineered)),
            ("columns", columns),
        ]
    )
    pipe.set_output(transform="pandas")
    return pipe


def build_selected_preprocessor(model_family: str = "tree") -> Pipeline:
    """Preprocessing with the feature set selected in issue #9 (the one to use from now on)."""
    return build_preprocessor(model_family, engineered=SELECTED_ENGINEERED, drop=SELECTED_DROP)


def save_preprocessor(pipe: Pipeline, path: Path) -> None:
    """Serialize a fitted pipeline (reused by the API)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load_preprocessor(path: Path) -> Pipeline:
    """Load a pipeline saved by save_preprocessor."""
    return joblib.load(Path(path))
