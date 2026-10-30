"""Hyperparameter tuning of the chosen model with Optuna (issue #13).

The objective is the mean PR AUC of a 5-fold stratified cross-validation on the
TRAINING set, with the same folds as every earlier comparison (seed 42). The
validation set plays no role in the search.

Search space (see SPACE) is deliberately modest: the first sanity check showed
that a bigger model (600 trees, 96 leaves) is worse than the default.
"""
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.models.cv import N_SPLITS, RANDOM_STATE, cross_validate_model
from src.models.zoo import build_model

SPACE = {
    "n_estimators": ("int", 100, 600),
    "learning_rate": ("float, log scale", 0.02, 0.2),
    "num_leaves": ("int, log scale", 8, 96),
    "min_child_samples": ("int", 5, 80),
    "subsample": ("float", 0.6, 1.0),
    "colsample_bytree": ("float", 0.5, 1.0),
    "reg_alpha": ("float, log scale", 1e-3, 10.0),
    "reg_lambda": ("float, log scale", 1e-3, 10.0),
    "class_weight": ("categorical", [None, "balanced"]),
}
FIXED = {"subsample_freq": 1}  # makes `subsample` effective in LightGBM
# Starting point close to the default configuration (reg_* cannot be 0 on a log scale).
START_POINT = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 1e-3,
    "reg_lambda": 1e-3,
    "class_weight": None,
}


def suggest_params(trial: optuna.Trial) -> dict:
    """Draw one configuration from SPACE."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 96, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def build_tuned_model(params: dict, preprocessing: str = "selected", random_state: int = RANDOM_STATE, n_jobs: int = 1):
    """Pipeline of the chosen LightGBM with the given hyperparameters."""
    model = build_model("lightgbm", preprocessing, random_state=random_state, n_jobs=n_jobs)
    model.set_params(**{f"model__{k}": v for k, v in {**FIXED, **params}.items()})
    return model


def run_study(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 40,
    seed: int = RANDOM_STATE,
    n_splits: int = N_SPLITS,
    cv_seed: int = RANDOM_STATE,
) -> optuna.Study:
    """Maximize the mean cross-validated PR AUC with a seeded TPE sampler."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        scores = cross_validate_model(build_tuned_model(params), X, y, n_splits=n_splits, random_state=cv_seed)
        trial.set_user_attr("pr_auc_std", float(scores["pr_auc"].std(ddof=1)))
        trial.set_user_attr("roc_auc", float(scores["roc_auc"].mean()))
        return float(scores["pr_auc"].mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(START_POINT)
    study.optimize(objective, n_trials=n_trials)
    return study


def trials_frame(study: optuna.Study) -> pd.DataFrame:
    """One row per trial: number, PR AUC mean/std, ROC AUC and the sampled parameters."""
    rows = []
    for t in study.trials:
        rows.append(
            {
                "trial": t.number,
                "pr_auc": t.value,
                "pr_auc_std": t.user_attrs.get("pr_auc_std"),
                "roc_auc": t.user_attrs.get("roc_auc"),
                **t.params,
            }
        )
    return pd.DataFrame(rows)


def save_results(study: optuna.Study, out_dir: Path) -> None:
    """Write the trial history (CSV) and the best parameters (JSON)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_frame(study).to_csv(out_dir / "lightgbm_trials.csv", index=False)
    payload = {"best_value_pr_auc": study.best_value, "best_trial": study.best_trial.number, "params": study.best_params}
    (out_dir / "lightgbm_best_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_best_params(path: Path) -> dict:
    """Best parameters saved by save_results."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["params"]
