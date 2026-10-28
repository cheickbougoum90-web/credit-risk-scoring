"""Model constructors for the comparison of issue #10.

Every model is a full pipeline (preprocessing + estimator). Hyperparameters are
deliberately close to the libraries' defaults: tuning is issue #13, and class
imbalance is handled in issue #11, so neither is done here.

preprocessing="selected": feature set chosen in issue #9 (recomputed ratio replaces
                          the leaky provided one).
preprocessing="original": all original columns, including loan_percent_income. Used only
                          for leakage probes, never for a final model.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features.preprocessing import build_preprocessor, build_selected_preprocessor

RANDOM_STATE = 42
MODEL_NAMES = ("logistic", "decision_tree", "random_forest", "xgboost", "lightgbm")
LINEAR_MODELS = ("logistic",)


def _estimator(name: str, random_state: int, n_jobs: int):
    if name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if name == "decision_tree":  # diagnostic reference: how far do a few simple rules go?
        return DecisionTreeClassifier(max_depth=5, random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=n_jobs, random_state=random_state)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            num_leaves=31,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def build_model(
    name: str, preprocessing: str = "selected", random_state: int = RANDOM_STATE, n_jobs: int = -1
) -> Pipeline:
    """Unfitted pipeline for one of MODEL_NAMES."""
    if preprocessing not in ("selected", "original"):
        raise ValueError("preprocessing must be 'selected' or 'original'")
    estimator = _estimator(name, random_state, n_jobs)
    family = "linear" if name in LINEAR_MODELS else "tree"
    prep = build_selected_preprocessor(family) if preprocessing == "selected" else build_preprocessor(family)
    return Pipeline([("preprocess", prep), ("model", estimator)])
