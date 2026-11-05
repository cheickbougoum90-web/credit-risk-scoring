"""SHAP explanations of the final model (issue #17), reusable by the API.

SHAP splits the model output for ONE customer into a base value plus one contribution per
feature. Everything is in log-odds, the scale on which a LightGBM classifier adds its trees:

    probability = sigmoid(base_value + sum of the contributions)

A positive contribution pushes the default probability up, a negative one pushes it down.
Contributions of the one-hot columns of a variable (and of the missing-value flag of the
employment length) are summed back to the ORIGINAL variable so that explanations read like
the application form: "loan grade", not "loan_grade" plus eleven dummy columns.
"""
import warnings

import numpy as np
import pandas as pd
import shap

LABELS = {
    "person_age": "Age",
    "person_income": "Annual income",
    "person_emp_length": "Employment length",
    "person_home_ownership": "Home ownership",
    "loan_intent": "Loan purpose",
    "loan_grade": "Loan grade",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_to_income": "Loan-to-income ratio",
    "cb_person_default_on_file": "Past default on file",
    "cb_person_cred_hist_length": "Credit history length",
}
_ONE_HOT_PREFIXES = ("person_home_ownership_", "loan_intent_")


def original_variable(column: str) -> str:
    """The variable of the application form that a transformed column comes from."""
    for prefix in _ONE_HOT_PREFIXES:
        if column.startswith(prefix):
            return prefix[:-1]
    if column == "person_emp_length_missing":
        return "person_emp_length"
    return column


def raw_value(X: pd.DataFrame, variable: str) -> pd.Series:
    """Value shown to a human for a variable (the loan-to-income ratio is derived from two columns)."""
    if variable == "loan_to_income":
        return X["loan_amnt"] / X["person_income"]
    return X[variable]


def _format(value) -> str:
    """Human-readable value: thousands separators, no useless decimals, 'missing' for absent values."""
    if pd.isna(value):
        return "missing"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if abs(value) < 10:
            return f"{value:.3g}"
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


class Explainer:
    """TreeExplainer around a fitted (preprocessing + LightGBM) pipeline. Build it once, reuse it."""

    def __init__(self, pipeline):
        steps = getattr(pipeline, "named_steps", {})
        if "preprocess" not in steps or "model" not in steps:
            raise TypeError("Explainer needs a fitted pipeline with 'preprocess' and 'model' steps (no calibrator)")
        self.pipeline = pipeline
        self._explainer = shap.TreeExplainer(steps["model"])

    def features(self, X: pd.DataFrame) -> pd.DataFrame:
        """The model inputs (after preprocessing) for the given raw customers."""
        return self.pipeline.named_steps["preprocess"].transform(X)

    def contributions(self, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """SHAP contributions per transformed column (log-odds) and the base value."""
        F = self.features(X)
        with warnings.catch_warnings():                  # shap warns that LightGBM binary output is a list
            warnings.simplefilter("ignore", UserWarning)
            values = self._explainer.shap_values(F)
        if isinstance(values, list):                     # older shap: one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:                             # newer shap: (rows, features, classes)
            values = values[:, :, 1]
        base = np.atleast_1d(np.asarray(self._explainer.expected_value, dtype=float))
        return pd.DataFrame(values, columns=F.columns, index=X.index), float(base[-1])

    def grouped(self, contributions: pd.DataFrame) -> pd.DataFrame:
        """Contributions summed per original variable."""
        mapping = {c: original_variable(c) for c in contributions.columns}
        return contributions.T.groupby(mapping).sum().T

    def top_factors(self, X: pd.DataFrame, k: int = 3) -> list:
        """For each customer, the k variables with the largest absolute contribution."""
        grouped = self.grouped(self.contributions(X)[0])
        out = []
        for i in range(len(X)):
            row = grouped.iloc[i]
            names = row.abs().sort_values(ascending=False).index[:k]
            factors = []
            for name in names:
                contribution = float(row[name])
                factors.append(
                    {
                        "feature": name,
                        "label": LABELS.get(name, name),
                        "value": _format(raw_value(X.iloc[[i]], name).iloc[0]),
                        "contribution": contribution,
                        "effect": "increases risk" if contribution > 0 else "decreases risk",
                    }
                )
            out.append(factors)
        return out

    def probability(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))
