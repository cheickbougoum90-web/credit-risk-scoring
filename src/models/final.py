"""The model used from issue #14 on (threshold, evaluation, explanations, API).

Single source of truth: notebooks and the API import it from here.

Choice history
--------------
Issue 10: LightGBM and XGBoost are equivalent in cross-validation; LightGBM is chosen.
Issue 13: Optuna tuning gave no gain beyond noise on folds not used by the search, so the
          default hyperparameters are kept.
Issue 14: neither sigmoid nor isotonic calibration beats the Brier score of the
          uncalibrated model beyond noise (isotonic also hurts the ranking), so no
          calibrator is applied.
"""
from src.models.calibration import build_calibrated_model
from src.models.zoo import RANDOM_STATE

FINAL_MODEL_NAME = "lightgbm"
FINAL_CALIBRATION = "none"


def build_final_model(n_jobs: int = -1, random_state: int = RANDOM_STATE):
    """Unfitted final pipeline: selected preprocessing + default LightGBM (+ calibrator if any)."""
    return build_calibrated_model(
        FINAL_CALIBRATION, FINAL_MODEL_NAME, random_state=random_state, n_jobs=n_jobs
    )
