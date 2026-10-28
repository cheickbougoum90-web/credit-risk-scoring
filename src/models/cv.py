"""Stratified cross-validation helpers (issues #9 to #11).

All folds are built from the TRAINING set only. Using the same folds
(same seed) for every candidate makes the fold-by-fold differences paired,
which is more informative than comparing two means.
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.models.evaluate import classification_metrics

CV_METRICS = ("roc_auc", "pr_auc", "ks", "brier")
N_SPLITS = 5
RANDOM_STATE = 42


def cross_validate_model(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    record_time: bool = False,
    extra: dict | None = None,
) -> pd.DataFrame:
    """Return one row of metrics per fold. The estimator is cloned and refitted on each fold.

    With record_time=True a `fit_seconds` column is added (not reproducible, hence optional).
    `extra` maps a column name to a function (y_true, scores) -> float computed on each validation fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        t0 = time.perf_counter()
        model = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        elapsed = time.perf_counter() - t0
        scores = model.predict_proba(X.iloc[va])[:, 1]
        m = classification_metrics(y.iloc[va], scores)
        row = {"fold": fold, **{k: m[k] for k in CV_METRICS}}
        for name, fn in (extra or {}).items():
            row[name] = float(fn(y.iloc[va], scores))
        if record_time:
            row["fit_seconds"] = elapsed
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation (ddof=1) of each metric across folds."""
    cols = [c for c in CV_METRICS if c in scores.columns]
    return pd.DataFrame({"mean": scores[cols].mean(), "std": scores[cols].std(ddof=1)})


def paired_gain(baseline: pd.DataFrame, candidate: pd.DataFrame, metric: str = "pr_auc") -> dict:
    """Fold-by-fold improvement of `candidate` over `baseline` (same folds required).

    A positive gain always means 'better' (the sign is flipped for the Brier score).
    """
    if len(baseline) != len(candidate) or not (baseline["fold"].values == candidate["fold"].values).all():
        raise ValueError("baseline and candidate must come from the same folds")
    sign = -1.0 if metric == "brier" else 1.0
    diff = sign * (candidate[metric].values - baseline[metric].values)
    return {
        "mean_gain": float(diff.mean()),
        "std_gain": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
        "folds_improved": int((diff > 0).sum()),
        "n_folds": int(len(diff)),
    }


def cv_comparison_table(results: dict) -> pd.DataFrame:
    """One row per model with the mean and sample std of each metric over the folds.

    `results` maps a model name to the DataFrame returned by cross_validate_model.
    Rows are sorted by decreasing mean PR AUC.
    """
    rows = {}
    for name, scores in results.items():
        s = summarize(scores)
        row = {}
        for metric in s.index:
            row[f"{metric}_mean"] = s.loc[metric, "mean"]
            row[f"{metric}_std"] = s.loc[metric, "std"]
        if "fit_seconds" in scores.columns:
            row["fit_seconds_mean"] = scores["fit_seconds"].mean()
        rows[name] = row
    table = pd.DataFrame(rows).T
    return table.sort_values("pr_auc_mean", ascending=False)


def cv_table_to_markdown(table: pd.DataFrame) -> str:
    """Markdown table of 'mean +/- std' for the main metrics (for the README)."""
    lines = [
        "| Model | PR AUC | ROC AUC | KS | Brier |",
        "|---|---|---|---|---|",
    ]
    for name, r in table.iterrows():
        cells = [f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_std']:.3f}" for m in ("pr_auc", "roc_auc", "ks", "brier")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
