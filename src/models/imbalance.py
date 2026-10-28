"""Class-imbalance strategies (issue #11).

strategy="none":          the model as is.
strategy="class_weight":  'balanced' class weights (a fold-independent setting).
strategy="smote":         SMOTE-NC oversampling of the minority class, applied ONLY to the
                          training part of each fold: the pipeline resamples in `fit` and
                          never in `predict`, so validation rows are never resampled.

SMOTE-NC is applied after preprocessing. One-hot columns and binary flags are treated as
categorical (the synthetic rows copy the majority value of the neighbours, so they stay 0/1);
the ordinal loan_grade and the numeric columns are interpolated.
"""
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator

from src.models.zoo import RANDOM_STATE, build_model

IMBALANCE_STRATEGIES = ("none", "class_weight", "smote")
CATEGORICAL_PREFIXES = ("person_home_ownership_", "loan_intent_")
CATEGORICAL_NAMES = ("person_emp_length_missing", "cb_person_default_on_file")


class SmoteNCByName(BaseEstimator):
    """SMOTE-NC that finds its categorical columns by name at fit time."""

    def __init__(self, k_neighbors: int = 5, random_state: int = RANDOM_STATE):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X: pd.DataFrame, y):
        cats = [
            i
            for i, c in enumerate(X.columns)
            if c.startswith(CATEGORICAL_PREFIXES) or c in CATEGORICAL_NAMES
        ]
        sampler = SMOTENC(
            categorical_features=cats, k_neighbors=self.k_neighbors, random_state=self.random_state
        )
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=getattr(y, "name", None))


def build_imbalance_model(
    model_name: str, strategy: str = "none", random_state: int = RANDOM_STATE, n_jobs: int = -1
) -> ImbPipeline:
    """Unfitted preprocessing + (optional resampler) + estimator pipeline."""
    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(f"strategy must be one of {IMBALANCE_STRATEGIES}, got {strategy!r}")
    base = build_model(model_name, "selected", random_state=random_state, n_jobs=n_jobs)
    prep, est = base.named_steps["preprocess"], base.named_steps["model"]

    if strategy == "class_weight":
        if "class_weight" not in est.get_params():
            raise ValueError(f"{model_name} has no class_weight parameter (XGBoost needs a fold-dependent scale_pos_weight)")
        est.set_params(class_weight="balanced")

    resampler = SmoteNCByName(random_state=random_state) if strategy == "smote" else "passthrough"
    # imblearn does not accept a nested Pipeline as an intermediate step: flatten the preprocessing steps.
    return ImbPipeline([*prep.steps, ("resample", resampler), ("model", est)])
