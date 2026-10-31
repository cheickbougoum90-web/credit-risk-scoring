import numpy as np
import pytest

from src.features.preprocessing import get_xy
from src.models.final import FINAL_CALIBRATION, FINAL_MODEL_NAME, build_final_model
from src.models.zoo import MODEL_NAMES, build_model
from tests.test_baseline import synthetic

pytest.importorskip("lightgbm")


def test_final_choices_are_valid_names():
    assert FINAL_MODEL_NAME in MODEL_NAMES
    assert FINAL_CALIBRATION in ("none", "sigmoid", "isotonic")


def test_final_model_fits_and_outputs_probabilities():
    X, y = get_xy(synthetic(n=1500, seed=4))
    p = build_final_model(n_jobs=1).fit(X.iloc[:1000], y.iloc[:1000]).predict_proba(X.iloc[1000:])[:, 1]
    assert p.shape == (500,) and ((p >= 0) & (p <= 1)).all()


def test_without_calibrator_the_final_model_is_the_plain_selected_model():
    X, y = get_xy(synthetic(n=1500, seed=4))
    plain = build_model(FINAL_MODEL_NAME, "selected", n_jobs=1).fit(X.iloc[:1000], y.iloc[:1000]).predict_proba(X.iloc[1000:])
    final = build_final_model(n_jobs=1).fit(X.iloc[:1000], y.iloc[:1000]).predict_proba(X.iloc[1000:])
    if FINAL_CALIBRATION == "none":
        assert np.allclose(plain, final)
