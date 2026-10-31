import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import get_xy
from src.models.calibration import CALIBRATION_METHODS, build_calibrated_model, calibrate
from src.models.evaluate import (
    calibration_slope_intercept,
    calibration_table,
    expected_calibration_error,
    safe_log_loss,
)
from src.models.imbalance import build_imbalance_model
from tests.test_baseline import synthetic

pytest.importorskip("lightgbm")


@pytest.fixture(scope="module")
def data():
    X, y = get_xy(synthetic(n=3000, seed=2))
    return X.iloc[:2000], y.iloc[:2000], X.iloc[2000:], y.iloc[2000:]


# ---------------- metrics ----------------
def test_ece_is_zero_when_bin_rates_match_and_known_value_otherwise():
    y = np.array([0, 0, 0, 0, 1] * 10)               # observed rate 0.2
    assert expected_calibration_error(y, np.full(50, 0.2), n_bins=5) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(50, 0.5), n_bins=5) == pytest.approx(0.3)


def test_calibration_table_bins_are_equal_sized_and_ordered():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    t = calibration_table(y, p, n_bins=10)
    assert t["n"].tolist() == [100] * 10
    assert t["mean_predicted"].is_monotonic_increasing
    assert (t["mean_predicted"] - t["observed_rate"]).abs().max() < 0.12  # calibrated by construction


def test_slope_is_one_for_calibrated_probabilities_and_below_one_for_overconfident_ones():
    rng = np.random.default_rng(1)
    p = np.clip(rng.beta(2, 5, 40000), 0.01, 0.99)
    y = (rng.random(40000) < p).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.08) and intercept == pytest.approx(0.0, abs=0.08)
    logit = np.log(p / (1 - p))
    over = 1 / (1 + np.exp(-2.5 * logit))            # same ranking, too extreme
    assert calibration_slope_intercept(y, over)[0] < 0.6


def test_log_loss_is_finite_even_for_hard_zero_and_one_probabilities():
    assert np.isfinite(safe_log_loss([0, 1, 1, 0], [0.0, 1.0, 0.0, 1.0]))


# ---------------- calibrators ----------------
@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_every_method_gives_valid_probabilities(method, data):
    Xa, ya, Xb, _ = data
    p = build_calibrated_model(method, n_jobs=1).fit(Xa, ya).predict_proba(Xb)[:, 1]
    assert p.shape == (len(Xb),) and ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_repairs_probabilities_distorted_by_class_weights(method, data):
    Xa, ya, Xb, yb = data
    raw = build_imbalance_model("logistic", "class_weight", n_jobs=1).fit(Xa, ya).predict_proba(Xb)[:, 1]
    cal = calibrate(build_imbalance_model("logistic", "class_weight", n_jobs=1), method).fit(Xa, ya).predict_proba(Xb)[:, 1]
    assert abs(cal.mean() - yb.mean()) < abs(raw.mean() - yb.mean()) / 2
    assert expected_calibration_error(yb, cal) < expected_calibration_error(yb, raw)


def test_calibration_keeps_the_ranking_for_sigmoid(data):
    from sklearn.metrics import roc_auc_score

    Xa, ya, Xb, yb = data
    raw = build_calibrated_model("none", n_jobs=1).fit(Xa, ya).predict_proba(Xb)[:, 1]
    sig = build_calibrated_model("sigmoid", n_jobs=1).fit(Xa, ya).predict_proba(Xb)[:, 1]
    assert abs(roc_auc_score(yb, sig) - roc_auc_score(yb, raw)) < 0.02


def test_calibrators_are_deterministic_and_validate_their_arguments(data):
    Xa, ya, Xb, _ = data
    a = build_calibrated_model("isotonic", n_jobs=1).fit(Xa, ya).predict_proba(Xb)
    b = build_calibrated_model("isotonic", n_jobs=1).fit(Xa, ya).predict_proba(Xb)
    assert np.allclose(a, b)
    with pytest.raises(ValueError):
        build_calibrated_model("platt")
    with pytest.raises(ValueError):
        calibrate(build_calibrated_model("none", n_jobs=1), "none")
