import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.features.preprocessing import get_xy
from src.models.zoo import MODEL_NAMES, build_model
from tests.test_baseline import synthetic

pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")


@pytest.fixture(scope="module")
def data():
    # 2,400 rows: smaller tables are too noisy for a depth-5 decision tree to generalize
    X, y = get_xy(synthetic(n=2400, seed=1))
    return X.iloc[:1600], y.iloc[:1600], X.iloc[1600:], y.iloc[1600:]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_outputs_valid_probabilities(name, data):
    Xa, ya, Xb, yb = data
    model = build_model(name, n_jobs=1).fit(Xa, ya)
    p = model.predict_proba(Xb)[:, 1]
    assert p.shape == (len(Xb),) and ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(yb, p) > 0.6  # a learnable signal is found


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_are_deterministic(name, data):
    Xa, ya, Xb, _ = data
    a = build_model(name, n_jobs=1).fit(Xa, ya).predict_proba(Xb)
    b = build_model(name, n_jobs=1).fit(Xa, ya).predict_proba(Xb)
    assert np.allclose(a, b)


def test_selected_and_original_preprocessing_differ_by_the_leaky_ratio(data):
    Xa, ya, _, _ = data
    sel = build_model("random_forest", "selected", n_jobs=1).named_steps["preprocess"].fit_transform(Xa)
    orig = build_model("random_forest", "original", n_jobs=1).named_steps["preprocess"].fit_transform(Xa)
    assert "loan_to_income" in sel.columns and "loan_percent_income" not in sel.columns
    assert "loan_percent_income" in orig.columns and "loan_to_income" not in orig.columns


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        build_model("neural_net")
    with pytest.raises(ValueError):
        build_model("logistic", preprocessing="everything")
