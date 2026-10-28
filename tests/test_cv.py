import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import get_xy
from src.models.baseline import build_baseline
from src.models.cv import cross_validate_model, paired_gain, summarize
from tests.test_baseline import synthetic


def test_one_row_per_fold_with_expected_metrics():
    X, y = get_xy(synthetic(n=600))
    scores = cross_validate_model(build_baseline(), X, y, n_splits=4)
    assert scores["fold"].tolist() == [0, 1, 2, 3]
    assert {"roc_auc", "pr_auc", "ks", "brier"} <= set(scores.columns)
    assert (scores["roc_auc"] > 0.5).all()


def test_cross_validation_is_deterministic():
    X, y = get_xy(synthetic(n=600))
    a = cross_validate_model(build_baseline(), X, y, n_splits=3)
    b = cross_validate_model(build_baseline(), X, y, n_splits=3)
    pd.testing.assert_frame_equal(a, b)


def test_estimator_passed_in_is_not_fitted_in_place():
    X, y = get_xy(synthetic(n=400))
    est = build_baseline()
    cross_validate_model(est, X, y, n_splits=3)
    assert not hasattr(est.named_steps["model"], "coef_")


def test_summarize_uses_sample_standard_deviation():
    scores = pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.6, 0.7, 0.8]})
    s = summarize(scores)
    assert s.loc["pr_auc", "mean"] == pytest.approx(0.7)
    assert s.loc["pr_auc", "std"] == pytest.approx(0.1)


def test_paired_gain_counts_improved_folds_and_flips_brier():
    base = pd.DataFrame({"fold": [0, 1, 2, 3], "pr_auc": [0.5, 0.5, 0.5, 0.5], "brier": [0.2, 0.2, 0.2, 0.2]})
    cand = pd.DataFrame({"fold": [0, 1, 2, 3], "pr_auc": [0.6, 0.5, 0.4, 0.7], "brier": [0.1, 0.2, 0.3, 0.2]})
    g = paired_gain(base, cand, "pr_auc")
    assert g["mean_gain"] == pytest.approx(0.05) and g["folds_improved"] == 2 and g["n_folds"] == 4
    b = paired_gain(base, cand, "brier")
    assert b["mean_gain"] == pytest.approx(0.0) and b["folds_improved"] == 1  # lower brier is better


def test_paired_gain_requires_same_folds():
    a = pd.DataFrame({"fold": [0, 1], "pr_auc": [0.5, 0.5]})
    b = pd.DataFrame({"fold": [0, 2], "pr_auc": [0.5, 0.5]})
    with pytest.raises(ValueError):
        paired_gain(a, b)


def test_record_time_adds_a_column_only_when_asked():
    from src.models.cv import cross_validate_model

    X, y = get_xy(synthetic(n=400))
    assert "fit_seconds" not in cross_validate_model(build_baseline(), X, y, n_splits=3).columns
    timed = cross_validate_model(build_baseline(), X, y, n_splits=3, record_time=True)
    assert (timed["fit_seconds"] > 0).all()


def test_comparison_table_is_sorted_by_pr_auc_and_renders_markdown():
    from src.models.cv import cv_comparison_table, cv_table_to_markdown

    def scores(pr):
        return pd.DataFrame({"fold": [0, 1, 2], "roc_auc": [0.9] * 3, "pr_auc": pr, "ks": [0.6] * 3, "brier": [0.1] * 3})

    table = cv_comparison_table({"weak": scores([0.5, 0.6, 0.7]), "strong": scores([0.8, 0.8, 0.9])})
    assert table.index.tolist() == ["strong", "weak"]
    assert table.loc["weak", "pr_auc_std"] == pytest.approx(0.1)
    md = cv_table_to_markdown(table)
    assert md.splitlines()[0].startswith("| Model |") and "weak" in md and "±" in md
