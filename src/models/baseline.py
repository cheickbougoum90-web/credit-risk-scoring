"""Baseline model (issue #8): logistic regression on the linear preprocessing variant.

Deliberately plain: no class weights, no feature engineering, default
regularization. Every later model must beat it.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessing import build_preprocessor

RANDOM_STATE = 42


def build_baseline(C: float = 1.0, random_state: int = RANDOM_STATE) -> Pipeline:
    """Unfitted pipeline: linear preprocessing + logistic regression."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor("linear")),
            ("model", LogisticRegression(C=C, max_iter=1000, random_state=random_state)),
        ]
    )
