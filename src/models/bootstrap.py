"""Bootstrap confidence intervals on a held-out set (issue #15).

Rows are resampled with replacement and every model is scored on the SAME resampled
rows, so differences between models get a paired interval. The intervals only capture
sampling noise of the evaluation set: they do not account for the choices made while
looking at that set (model selection), nor for any shift between this set and real data.
"""
import numpy as np
import pandas as pd

from src.models.evaluate import classification_metrics, precision_at_recall

REPORTED_METRICS = ("roc_auc", "pr_auc", "ks", "brier", "recall", "precision", "precision_at_recall80")


def point_estimates(y_true, scores, threshold: float = 0.5, recall_target: float = 0.8) -> dict:
    """The reported metrics on the full evaluation set."""
    m = classification_metrics(y_true, scores, threshold)
    out = {k: m[k] for k in ("roc_auc", "pr_auc", "ks", "brier", "recall", "precision")}
    out["precision_at_recall80"] = precision_at_recall(y_true, scores, recall_target)
    return out


def bootstrap_metrics(
    y_true,
    scores: dict,
    n_boot: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
    recall_target: float = 0.8,
) -> dict:
    """Metrics of every model on `n_boot` common bootstrap resamples.

    `scores` maps a model name to its predicted probabilities. Returns a dict name -> DataFrame
    with one row per resample (rows aligned across models).
    """
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        raise ValueError("y_true must contain both classes")
    arrays = {name: np.asarray(s, dtype=float) for name, s in scores.items()}
    if any(a.shape != y.shape for a in arrays.values()):
        raise ValueError("every score array must have the same length as y_true")

    rng = np.random.default_rng(seed)
    rows = {name: [] for name in arrays}
    while len(next(iter(rows.values()))) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:  # a degenerate resample: draw again
            continue
        for name, s in arrays.items():
            rows[name].append(point_estimates(y[idx], s[idx], threshold, recall_target))
    return {name: pd.DataFrame(r) for name, r in rows.items()}


def percentile_interval(values, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap replicates."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    alpha = (1 - level) / 2
    lo, hi = np.quantile(np.asarray(values, dtype=float), [alpha, 1 - alpha])
    return float(lo), float(hi)


def summarize_intervals(replicates: pd.DataFrame, estimates: dict, level: float = 0.95) -> pd.DataFrame:
    """One row per metric: point estimate on the full set and percentile interval."""
    rows = {}
    for metric in REPORTED_METRICS:
        lo, hi = percentile_interval(replicates[metric], level)
        rows[metric] = {"estimate": estimates[metric], "ci_low": lo, "ci_high": hi}
    return pd.DataFrame(rows).T


def paired_difference(rep_a: pd.DataFrame, rep_b: pd.DataFrame, est_a: dict, est_b: dict, level: float = 0.95) -> pd.DataFrame:
    """Difference a - b per metric with a paired interval, and the share of resamples where a > b.

    For the Brier score a LOWER value is better, so read its sign accordingly.
    """
    rows = {}
    for metric in REPORTED_METRICS:
        diff = rep_a[metric].to_numpy() - rep_b[metric].to_numpy()
        lo, hi = percentile_interval(diff, level)
        rows[metric] = {
            "difference": est_a[metric] - est_b[metric],
            "ci_low": lo,
            "ci_high": hi,
            "share_a_higher": float((diff > 0).mean()),
        }
    return pd.DataFrame(rows).T
