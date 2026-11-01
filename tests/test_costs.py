import numpy as np
import pandas as pd
import pytest

from src.models.costs import (
    THRESHOLD_GRID,
    accept_all_cost,
    bootstrap_policy_costs,
    cost_curve,
    customer_costs,
    near_optimal_range,
    optimal_threshold,
    policy_cost,
    theoretical_thresholds,
)


def tiny():
    # four customers: amounts and rates chosen to make the arithmetic obvious
    amount = np.array([1000.0, 2000.0, 1000.0, 4000.0])
    rate = np.array([10.0, 20.0, 5.0, 10.0])
    y = np.array([1, 0, 1, 0])
    return amount, rate, y


def test_customer_costs_follow_the_cost_matrix():
    amount, rate, _ = tiny()
    fn, fp = customer_costs(amount, rate, lgd=0.5)
    assert fn.tolist() == [500.0, 1000.0, 500.0, 2000.0]
    assert fp.tolist() == [100.0, 400.0, 50.0, 400.0]


def test_policy_cost_counts_missed_defaults_and_refused_good_customers():
    amount, rate, y = tiny()
    fn, fp = customer_costs(amount, rate, lgd=0.5)
    # refuse customers 1 and 2 (indices 1,2): refusing a good one (index 1) costs fp=400,
    # accepting the default at index 0 costs fn=500, refusing the default at index 2 costs 0
    assert policy_cost(y, [False, True, True, False], fn, fp) == pytest.approx(900.0)
    assert policy_cost(y, [True, False, True, False], fn, fp) == 0.0          # perfect decisions
    assert accept_all_cost(y, fn) == pytest.approx(1000.0)                    # fn of the two defaults
    assert policy_cost(y, [False] * 4, fn, fp) == accept_all_cost(y, fn)


def test_curve_extremes_match_accept_all_and_refuse_all():
    amount, rate, y = tiny()
    fn, fp = customer_costs(amount, rate)
    curve = cost_curve(y, [0.9, 0.2, 0.8, 0.1], fn, fp, thresholds=[0.0, 1.01])
    assert curve.loc[0, "cost"] == pytest.approx(fp[y == 0].sum())             # threshold 0: everybody refused
    assert curve.loc[1, "cost"] == pytest.approx(accept_all_cost(y, fn))        # threshold above 1: nobody refused
    assert curve.loc[0, "refusal_rate"] == 1.0 and curve.loc[1, "refusal_rate"] == 0.0


def test_optimal_threshold_matches_theory_for_calibrated_probabilities():
    rng = np.random.default_rng(0)
    n = 200_000
    p = np.clip(rng.beta(1.2, 4.0, n), 0.001, 0.999)
    y = (rng.random(n) < p).astype(int)                 # perfectly calibrated by construction
    for ratio in (3.0, 5.0, 10.0):
        fn, fp = np.full(n, ratio), np.ones(n)
        t = optimal_threshold(cost_curve(y, p, fn, fp))
        assert t == pytest.approx(1 / (1 + ratio), abs=0.03)


def test_theoretical_threshold_formula_and_range_helper():
    assert theoretical_thresholds([5.0, 9.0], [1.0, 1.0]).tolist() == pytest.approx([1 / 6, 0.1])
    curve = pd.DataFrame({"threshold": [0.1, 0.2, 0.3, 0.4], "cost": [105.0, 100.0, 100.5, 120.0]})
    assert optimal_threshold(curve) == 0.2
    assert near_optimal_range(curve, tolerance=0.01) == (0.2, 0.3)


def test_bootstrap_costs_are_reproducible_and_paired():
    amount, rate, y = tiny()
    fn, fp = customer_costs(np.tile(amount, 50), np.tile(rate, 50))
    yy = np.tile(y, 50)
    refuse = {"a": np.tile([False, True, True, False], 50), "b": np.tile([True, False, True, False], 50)}
    r1 = bootstrap_policy_costs(yy, refuse, fn, fp, n_boot=40, seed=3)
    r2 = bootstrap_policy_costs(yy, refuse, fn, fp, n_boot=40, seed=3)
    pd.testing.assert_frame_equal(r1, r2)
    assert set(r1.columns) == {"accept_all", "a", "b"}
    assert (r1["b"] <= r1["a"]).all()        # the perfect policy is never worse on the same resample


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        customer_costs([1000.0], [10.0], lgd=0.0)
    with pytest.raises(ValueError):
        customer_costs([1000.0], [np.nan])
    assert THRESHOLD_GRID.min() > 0 and THRESHOLD_GRID.max() < 1
