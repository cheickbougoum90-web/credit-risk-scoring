import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.features.preprocessing import get_xy
from src.models.cv import cross_validate_model
from src.models.evaluate import precision_at_recall
from src.models.imbalance import IMBALANCE_STRATEGIES, SmoteNCByName, build_imbalance_model
from tests.test_baseline import synthetic

pytest.importorskip("lightgbm")


@pytest.fixture(scope="module")
def data():
    X, y = get_xy(synthetic(n=2400, seed=1))
    return X.iloc[:1600], y.iloc[:1600], X.iloc[1600:], y.iloc[1600:]


def encoded_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "loan_amnt": rng.normal(size=n),
            "loan_grade": rng.integers(0, 7, n).astype(float),
            "person_home_ownership_RENT": rng.integers(0, 2, n).astype(float),
            "person_home_ownership_OWN": rng.integers(0, 2, n).astype(float),
            "person_emp_length_missing": rng.integers(0, 2, n).astype(float),
        }
    )
    y = pd.Series((rng.random(n) < 0.15).astype(int), name="loan_status")
    return X, y


def test_smote_balances_classes_and_keeps_categorical_columns_binary():
    X, y = encoded_frame()
    Xr, yr = SmoteNCByName().fit_resample(X, y)
    counts = yr.value_counts()
    assert counts[0] == counts[1]
    assert len(Xr) > len(X)
    for col in ["person_home_ownership_RENT", "person_home_ownership_OWN", "person_emp_length_missing"]:
        assert set(Xr[col].unique()) <= {0.0, 1.0}  # no fractional one-hot values
    assert list(Xr.columns) == list(X.columns)


def test_smote_keeps_the_original_rows_unchanged():
    X, y = encoded_frame()
    Xr, _ = SmoteNCByName().fit_resample(X, y)
    pd.testing.assert_frame_equal(Xr.iloc[: len(X)].reset_index(drop=True), X.reset_index(drop=True), check_dtype=False)


@pytest.mark.parametrize("strategy", IMBALANCE_STRATEGIES)
@pytest.mark.parametrize("model", ["logistic", "lightgbm"])
def test_every_strategy_fits_and_predicts_on_unresampled_data(model, strategy, data):
    Xa, ya, Xb, _ = data
    pipe = build_imbalance_model(model, strategy, n_jobs=1).fit(Xa, ya)
    p = pipe.predict_proba(Xb)[:, 1]
    assert p.shape == (len(Xb),)  # validation rows are never resampled
    assert ((p >= 0) & (p <= 1)).all()


def test_class_weight_is_applied_and_smote_resamples_only_in_fit(data):
    Xa, ya, _, _ = data
    cw = build_imbalance_model("lightgbm", "class_weight", n_jobs=1)
    assert cw.named_steps["model"].get_params()["class_weight"] == "balanced"
    sm = build_imbalance_model("logistic", "smote", n_jobs=1)
    assert isinstance(sm.named_steps["resample"], SmoteNCByName)
    none = build_imbalance_model("logistic", "none", n_jobs=1)
    assert none.named_steps["resample"] == "passthrough"


def test_resampling_changes_the_mean_predicted_probability(data):
    Xa, ya, Xb, _ = data
    plain = build_imbalance_model("logistic", "none", n_jobs=1).fit(Xa, ya).predict_proba(Xb)[:, 1].mean()
    weighted = build_imbalance_model("logistic", "class_weight", n_jobs=1).fit(Xa, ya).predict_proba(Xb)[:, 1].mean()
    assert weighted > plain + 0.1  # probabilities are shifted upward by balancing


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        build_imbalance_model("logistic", "undersampling")
    with pytest.raises(ValueError):
        build_imbalance_model("xgboost", "class_weight")


def test_pipeline_is_clonable_and_works_with_cross_validation(data):
    Xa, ya, _, _ = data
    est = clone(build_imbalance_model("logistic", "smote", n_jobs=1))
    scores = cross_validate_model(est, Xa, ya, n_splits=3, extra={"p80": precision_at_recall})
    assert len(scores) == 3 and (scores["p80"] > 0).all()


def test_precision_at_recall_known_values():
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    assert precision_at_recall(y, s, 0.5) == pytest.approx(1.0)   # top score is a positive
    assert precision_at_recall(y, s, 1.0) == pytest.approx(2 / 3)  # both positives need the top 3 scores: 2 hits out of 3
    with pytest.raises(ValueError):
        precision_at_recall(y, s, 0.0)


def test_strategy_none_gives_the_same_predictions_as_the_plain_model(data):
    from src.models.zoo import build_model

    Xa, ya, Xb, _ = data
    plain = build_model("logistic", "selected", n_jobs=1).fit(Xa, ya).predict_proba(Xb)
    flat = build_imbalance_model("logistic", "none", n_jobs=1).fit(Xa, ya).predict_proba(Xb)
    assert np.allclose(plain, flat)  # flattening the preprocessing steps changed nothing
