import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import get_xy
from src.models.explain import LABELS, Explainer, _format, original_variable, sigmoid
from src.models.final import build_final_model
from tests.test_baseline import synthetic

pytest.importorskip("lightgbm")
pytest.importorskip("shap")


@pytest.fixture(scope="module")
def fitted():
    X, y = get_xy(synthetic(n=2400, seed=5))
    model = build_final_model(n_jobs=1).fit(X.iloc[:1800], y.iloc[:1800])
    return Explainer(model), X.iloc[1800:], y.iloc[1800:]


def test_contributions_add_up_to_the_model_probability(fitted):
    exp, X, _ = fitted
    contrib, base = exp.contributions(X)
    rebuilt = sigmoid(base + contrib.sum(axis=1).to_numpy())
    assert np.allclose(rebuilt, exp.probability(X), atol=1e-6)     # SHAP is an exact decomposition


def test_grouping_keeps_the_total_and_uses_original_variables(fitted):
    exp, X, _ = fitted
    contrib, _ = exp.contributions(X)
    grouped = exp.grouped(contrib)
    assert np.allclose(grouped.sum(axis=1), contrib.sum(axis=1))
    assert set(grouped.columns) <= set(LABELS)                      # every explained variable has a readable label
    assert not any(c.startswith(("person_home_ownership_", "loan_intent_")) for c in grouped.columns)


def test_original_variable_mapping():
    assert original_variable("person_home_ownership_RENT") == "person_home_ownership"
    assert original_variable("loan_intent_MEDICAL") == "loan_intent"
    assert original_variable("person_emp_length_missing") == "person_emp_length"
    assert original_variable("loan_grade") == "loan_grade"


def test_top_factors_are_sorted_signed_and_readable(fitted):
    exp, X, _ = fitted
    factors = exp.top_factors(X.iloc[:5], k=3)
    assert len(factors) == 5 and all(len(f) == 3 for f in factors)
    for per_customer in factors:
        sizes = [abs(f["contribution"]) for f in per_customer]
        assert sizes == sorted(sizes, reverse=True)
        for f in per_customer:
            assert f["effect"] == ("increases risk" if f["contribution"] > 0 else "decreases risk")
            assert f["label"] and isinstance(f["value"], str)


def test_missing_values_are_displayed_as_missing(fitted):
    exp, X, _ = fitted
    row = X.iloc[[0]].copy()
    row["person_emp_length"] = np.nan
    row["loan_int_rate"] = np.nan
    all_factors = exp.top_factors(row, k=11)[0]
    shown = {f["feature"]: f["value"] for f in all_factors}
    assert shown["person_emp_length"] == "missing" and shown["loan_int_rate"] == "missing"


def test_strong_drivers_of_the_synthetic_signal_rank_first(fitted):
    exp, X, _ = fitted
    importance = exp.grouped(exp.contributions(X)[0]).abs().mean().sort_values(ascending=False)
    assert set(importance.index[:4]) & {"loan_int_rate", "loan_grade", "loan_to_income"}


def test_explaining_one_customer_is_fast_enough_for_an_api(fitted):
    import time

    exp, X, _ = fitted
    t = time.perf_counter()
    exp.top_factors(X.iloc[[0]], k=3)
    assert time.perf_counter() - t < 1.0


def test_explainer_rejects_a_calibrated_model(fitted):
    from src.models.calibration import build_calibrated_model

    _, X, y = fitted
    calibrated = build_calibrated_model("sigmoid", n_jobs=1).fit(X, y)
    with pytest.raises(TypeError):
        Explainer(calibrated)


def test_values_are_formatted_for_humans():
    assert _format(11.54) == "11.54"          # an interest rate keeps its decimals
    assert _format(60000) == "60,000" and _format(60000.0) == "60,000"
    assert _format(0.2333) == "0.233"
    assert _format(float("nan")) == "missing"
    assert _format("D") == "D"
