"""Cost-based decision threshold (issue #16).

Cost matrix (see the business problem in the README):
- accepting a loan that ends in default (false negative) loses a share of the principal:
      cost_fn = LGD * loan_amnt            (LGD = loss given default, 60% by default)
- refusing a customer who would have repaid (false positive) loses the interest income:
      cost_fp = loan_amnt * loan_int_rate / 100     (one-year proxy)
- accepting a good customer and refusing a defaulter cost nothing.

A customer is REFUSED when the predicted default probability is at least the threshold.
Costs are per customer, so a threshold is evaluated on the total cost of a portfolio.
"""
import numpy as np
import pandas as pd

DEFAULT_LGD = 0.6
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.005), 3)


def customer_costs(amount, rate_pct, lgd: float = DEFAULT_LGD) -> tuple[np.ndarray, np.ndarray]:
    """Per-customer (cost_fn, cost_fp) from the loan amount and the interest rate in percent."""
    if not 0 < lgd <= 1:
        raise ValueError("lgd must be in (0, 1]")
    amount = np.asarray(amount, dtype=float)
    rate = np.asarray(rate_pct, dtype=float)
    if np.isnan(rate).any():
        raise ValueError("rate_pct must not contain missing values (impute them first)")
    return lgd * amount, amount * rate / 100.0


def policy_cost(y_true, refuse, cost_fn, cost_fp) -> float:
    """Total cost of a decision policy: missed defaults plus wrongly refused good customers."""
    y = np.asarray(y_true).astype(bool)
    refuse = np.asarray(refuse).astype(bool)
    return float(np.asarray(cost_fp)[refuse & ~y].sum() + np.asarray(cost_fn)[~refuse & y].sum())


def accept_all_cost(y_true, cost_fn) -> float:
    """Cost of a policy without model: everybody is accepted, so every default is a loss."""
    return float(np.asarray(cost_fn)[np.asarray(y_true).astype(bool)].sum())


def cost_curve(y_true, scores, cost_fn, cost_fp, thresholds=THRESHOLD_GRID) -> pd.DataFrame:
    """Total cost, refusal rate, recall and false-alarm rate for each threshold."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    rows = []
    for t in thresholds:
        refuse = s >= t
        rows.append(
            {
                "threshold": float(t),
                "cost": policy_cost(y, refuse, cost_fn, cost_fp),
                "refusal_rate": float(refuse.mean()),
                "recall": float((refuse & y).sum() / max(y.sum(), 1)),
                "false_alarm_rate": float((refuse & ~y).sum() / max((~y).sum(), 1)),
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(curve: pd.DataFrame) -> float:
    """Threshold with the lowest total cost (the lowest one in case of a tie)."""
    return float(curve.loc[curve["cost"].idxmin(), "threshold"])


def near_optimal_range(curve: pd.DataFrame, tolerance: float = 0.01) -> tuple[float, float]:
    """Smallest and largest thresholds whose cost is within `tolerance` (relative) of the minimum."""
    best = curve["cost"].min()
    ok = curve.loc[curve["cost"] <= best * (1 + tolerance), "threshold"]
    return float(ok.min()), float(ok.max())


def theoretical_thresholds(cost_fn, cost_fp) -> np.ndarray:
    """Customer-specific cut-off cost_fp / (cost_fp + cost_fn).

    It is optimal for each customer IF the probabilities are perfectly calibrated.
    """
    cost_fn, cost_fp = np.asarray(cost_fn, dtype=float), np.asarray(cost_fp, dtype=float)
    return cost_fp / (cost_fp + cost_fn)


def bootstrap_policy_costs(y_true, refuse_by_policy: dict, cost_fn, cost_fp, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Total cost of every fixed policy (plus accept-all) on common bootstrap resamples of the customers."""
    y = np.asarray(y_true).astype(bool)
    cost_fn, cost_fp = np.asarray(cost_fn, dtype=float), np.asarray(cost_fp, dtype=float)
    refuse_by_policy = {k: np.asarray(v).astype(bool) for k, v in refuse_by_policy.items()}
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        row = {"accept_all": float(cost_fn[idx][y[idx]].sum())}
        for name, refuse in refuse_by_policy.items():
            row[name] = policy_cost(y[idx], refuse[idx], cost_fn[idx], cost_fp[idx])
        rows.append(row)
    return pd.DataFrame(rows)
