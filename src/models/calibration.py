"""Probability calibration (issue #14).

The calibrator is learned on out-of-fold predictions of the base model, so it never
sees predictions made on the rows the base model was trained on.
`ensemble=False` keeps a single base model refitted on all the training data plus one
calibrator, which is simple to serialize and deploy.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from src.models.zoo import RANDOM_STATE, build_model

CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")


def calibrate(estimator, method: str, cv_splits: int = 5, random_state: int = RANDOM_STATE):
    """Wrap any probabilistic estimator in a cross-fitted calibrator."""
    if method not in ("sigmoid", "isotonic"):
        raise ValueError("method must be 'sigmoid' or 'isotonic'")
    folds = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    return CalibratedClassifierCV(estimator, method=method, cv=folds, ensemble=False)


def build_calibrated_model(
    method: str = "none", model_name: str = "lightgbm", random_state: int = RANDOM_STATE, n_jobs: int = -1
):
    """Selected-features model, optionally calibrated ('none', 'sigmoid' or 'isotonic')."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"method must be one of {CALIBRATION_METHODS}, got {method!r}")
    base = build_model(model_name, "selected", random_state=random_state, n_jobs=n_jobs)
    return base if method == "none" else calibrate(base, method, random_state=random_state)
