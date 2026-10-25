import numpy as np
import pandas as pd

from src.features.preprocessing import get_xy
from src.models.baseline import build_baseline


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    grade = rng.choice(list("ABCDEFG"), n, p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02])
    rate = 7 + 2 * np.array([ord(g) - 65 for g in grade]) + rng.normal(0, 1, n)
    pct = rng.uniform(0.02, 0.5, n)
    logit = -3 + 0.25 * (rate - 11) + 5 * (pct - 0.17)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "person_age": rng.integers(20, 60, n),
            "person_income": rng.lognormal(10.8, 0.5, n),
            "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
            "person_emp_length": np.where(rng.random(n) < 0.05, np.nan, rng.integers(0, 15, n).astype(float)),
            "loan_intent": rng.choice(["EDUCATION", "MEDICAL", "VENTURE"], n),
            "loan_grade": grade,
            "loan_amnt": rng.integers(500, 20000, n).astype(float),
            "loan_int_rate": np.where(rng.random(n) < 0.1, np.nan, rate),
            "loan_percent_income": pct,
            "cb_person_default_on_file": rng.choice(["N", "Y"], n),
            "cb_person_cred_hist_length": rng.integers(2, 20, n),
            "loan_status": y,
        }
    )
    return df


def test_baseline_fits_and_outputs_probabilities():
    X, y = get_xy(synthetic())
    model = build_baseline().fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()


def test_baseline_beats_chance_on_a_learnable_signal():
    from sklearn.metrics import roc_auc_score

    X, y = get_xy(synthetic(n=1500))
    model = build_baseline().fit(X.iloc[:1000], y.iloc[:1000])
    auc = roc_auc_score(y.iloc[1000:], model.predict_proba(X.iloc[1000:])[:, 1])
    assert auc > 0.65


def test_baseline_is_deterministic():
    X, y = get_xy(synthetic())
    a = build_baseline().fit(X, y).predict_proba(X)
    b = build_baseline().fit(X, y).predict_proba(X)
    assert np.allclose(a, b)
