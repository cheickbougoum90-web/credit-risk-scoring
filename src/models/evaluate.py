"""Evaluation metrics and the model comparison table (issues #8, #15).

Threshold-independent metrics (ROC AUC, PR AUC, KS, Brier) are the primary
comparison criteria. Recall and precision are reported at a fixed threshold
(default 0.5) as a reference; the cost-optimal threshold is chosen later.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

COMPARISON_COLUMNS = [
    "model",
    "split",
    "roc_auc",
    "pr_auc",
    "ks",
    "brier",
    "recall",
    "precision",
    "threshold",
    "notes",
]
METRIC_COLUMNS = ["roc_auc", "pr_auc", "ks", "brier", "recall", "precision", "threshold"]


def _validate(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true and y_score must have the same shape, got {y_true.shape} and {y_score.shape}")
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both classes")
    return y_true, y_score


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov statistic: max gap between the score distributions of the two classes."""
    y_true, y_score = _validate(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Compute ranking, calibration and thresholded metrics for predicted default probabilities."""
    y_true, y_score = _validate(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def precision_at_recall(y_true, y_score, recall_target: float = 0.8) -> float:
    """Best precision reachable while keeping recall >= recall_target (threshold-free operating point)."""
    y_true, y_score = _validate(y_true, y_score)
    if not 0 < recall_target <= 1:
        raise ValueError("recall_target must be in (0, 1]")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(precision[recall >= recall_target].max())


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table with equal-size (quantile) bins: mean predicted vs observed default rate."""
    y_true, y_prob = _validate(y_true, y_prob)
    order = np.argsort(y_prob, kind="mergesort")
    rows = []
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "mean_predicted": float(y_prob[idx].mean()),
                "observed_rate": float(y_true[idx].mean()),
                "n": int(len(idx)),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE with equal-size bins: weighted mean gap between predicted and observed rates."""
    table = calibration_table(y_true, y_prob, n_bins)
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["mean_predicted"] - table["observed_rate"]).abs()).sum())


def calibration_slope_intercept(y_true, y_prob) -> tuple[float, float]:
    """Slope and intercept of a logistic regression of the outcome on logit(p).

    A perfectly calibrated model has slope 1 and intercept 0. A slope below 1 means the
    predictions are too extreme; the intercept captures a global over- or under-estimation.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    p = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def safe_log_loss(y_true, y_prob) -> float:
    """Log loss with probabilities clipped away from 0 and 1."""
    y_true, y_prob = _validate(y_true, y_prob)
    return float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1]))


def update_comparison_table(
    path: Path, model_name: str, metrics: dict, split: str = "validation", notes: str = ""
) -> pd.DataFrame:
    """Add or replace the (model, split) row of the comparison CSV and return the table."""
    path = Path(path)
    row = {"model": model_name, "split": split, "notes": notes}
    row.update({k: round(float(metrics[k]), 4) for k in METRIC_COLUMNS})
    new = pd.DataFrame([row], columns=COMPARISON_COLUMNS)

    if path.exists():
        table = pd.read_csv(path)
        keep = ~((table["model"] == model_name) & (table["split"] == split))
        table = pd.concat([table.loc[keep], new], ignore_index=True)
    else:
        table = new
    path.parent.mkdir(parents=True, exist_ok=True)
    table[COMPARISON_COLUMNS].to_csv(path, index=False)
    return table[COMPARISON_COLUMNS]


def comparison_to_markdown(table: pd.DataFrame, split: str = "validation") -> str:
    """Render the rows of one split as a Markdown table (for the README)."""
    cols = ["model", "roc_auc", "pr_auc", "ks", "brier", "recall", "precision"]
    rows = table.loc[table["split"] == split, cols]
    header = "| Model | ROC AUC | PR AUC | KS | Brier | Recall | Precision |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows.itertuples(index=False):
        lines.append(f"| {r.model} | {r.roc_auc:.3f} | {r.pr_auc:.3f} | {r.ks:.3f} | {r.brier:.3f} | {r.recall:.3f} | {r.precision:.3f} |")
    return "\n".join(lines)
