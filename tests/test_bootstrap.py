import numpy as np
import pandas as pd
import pytest

from src.models.bootstrap import (
    REPORTED_METRICS,
    bootstrap_metrics,
    paired_difference,
    percentile_interval,
    point_estimates,
    summarize_intervals,
)


def make_scores(n, seed=0, signal=2.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    s = 1 / (1 + np.exp(-(signal * (y - 0.5) + rng.normal(0, 1, n) - 1)))
    return y, s


def test_point_estimates_has_all_reported_metrics():
    y, s = make_scores(500)
    assert set(point_estimates(y, s)) == set(REPORTED_METRICS)


def test_perfect_scores_give_degenerate_intervals_at_one():
    y = np.array([0] * 80 + [1] * 20)
    s = np.where(y == 1, 0.9, 0.1)
    rep = bootstrap_metrics(y, {"m": s}, n_boot=50)["m"]
    assert (rep["roc_auc"] == 1.0).all() and (rep["ks"] == 1.0).all()


def test_same_seed_reproduces_the_replicates_and_other_seed_does_not():
    y, s = make_scores(600)
    a = bootstrap_metrics(y, {"m": s}, n_boot=30, seed=1)["m"]
    b = bootstrap_metrics(y, {"m": s}, n_boot=30, seed=1)["m"]
    c = bootstrap_metrics(y, {"m": s}, n_boot=30, seed=2)["m"]
    pd.testing.assert_frame_equal(a, b)
    assert not a["roc_auc"].equals(c["roc_auc"])


def test_point_estimate_lies_inside_its_interval():
    y, s = make_scores(1500, seed=3)
    rep = bootstrap_metrics(y, {"m": s}, n_boot=300)["m"]
    table = summarize_intervals(rep, point_estimates(y, s))
    assert ((table["ci_low"] <= table["estimate"]) & (table["estimate"] <= table["ci_high"])).all()


def test_interval_narrows_with_more_data():
    width = {}
    for n in (400, 4000):
        y, s = make_scores(n, seed=5)
        rep = bootstrap_metrics(y, {"m": s}, n_boot=200)["m"]
        lo, hi = percentile_interval(rep["roc_auc"])
        width[n] = hi - lo
    assert width[4000] < width[400] / 2


def test_models_share_the_same_resamples_so_identical_models_have_zero_difference():
    y, s = make_scores(800, seed=7)
    reps = bootstrap_metrics(y, {"a": s, "b": s.copy()}, n_boot=40)
    diff = paired_difference(reps["a"], reps["b"], point_estimates(y, s), point_estimates(y, s))
    assert (diff["difference"] == 0).all() and (diff["ci_low"] == 0).all() and (diff["ci_high"] == 0).all()


def test_better_model_has_a_positive_paired_difference_interval():
    y, good = make_scores(2000, seed=9, signal=3.0)
    _, weak = make_scores(2000, seed=9, signal=0.5)
    reps = bootstrap_metrics(y, {"good": good, "weak": weak}, n_boot=200)
    diff = paired_difference(reps["good"], reps["weak"], point_estimates(y, good), point_estimates(y, weak))
    assert diff.loc["roc_auc", "ci_low"] > 0 and diff.loc["roc_auc", "share_a_higher"] == 1.0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        bootstrap_metrics([0, 0, 0], {"m": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError):
        bootstrap_metrics([0, 1, 0, 1], {"m": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError):
        percentile_interval([0.1, 0.2], level=1.5)
