import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    ENGINEERED_FEATURES,
    HIGH_BURDEN_THRESHOLD,
    SELECTED_DROP,
    SELECTED_ENGINEERED,
    FeatureEngineer,
)
from src.features.preprocessing import (
    build_preprocessor,
    build_selected_preprocessor,
    get_xy,
    load_preprocessor,
    save_preprocessor,
)
from tests.test_preprocessing import frame


def base_frame():
    # grade A rates: 10, 12 -> median 11 ; grade B rates: 14 -> median 14
    return pd.DataFrame(
        {
            "person_age": [20.0, 40.0, 25.0],
            "person_income": [40000.0, 50000.0, 20000.0],
            "loan_amnt": [4000.0, 10000.0, 6000.0],
            "loan_int_rate": [10.0, 12.0, 14.0],
            "loan_grade": ["A", "A", "B"],
            "loan_percent_income": [0.10, 0.25, 0.2501],
            "cb_person_cred_hist_length": [2.0, 10.0, 5.0],
        }
    )


def test_feature_values_are_computed_as_documented():
    X = base_frame()
    out = FeatureEngineer(ENGINEERED_FEATURES).fit(X).transform(X)
    assert out["loan_to_income"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert out["rate_vs_grade_median"].tolist() == pytest.approx([-1.0, 1.0, 0.0])
    assert out["cred_hist_to_age"].tolist() == pytest.approx([0.1, 0.25, 0.2])


def test_high_burden_threshold_is_strict():
    X = base_frame()
    out = FeatureEngineer(["high_burden"]).fit(X).transform(X)
    assert HIGH_BURDEN_THRESHOLD == 0.25
    assert out["high_burden"].tolist() == [0.0, 0.0, 1.0]  # 0.25 is not above the threshold


def test_grade_medians_come_from_fit_data_only():
    train = base_frame()
    other = base_frame().assign(loan_int_rate=[30.0, 30.0, 30.0], loan_grade=["A", "A", "G"])
    eng = FeatureEngineer(["rate_vs_grade_median"]).fit(train)
    out = eng.transform(other)
    assert out["rate_vs_grade_median"].tolist() == pytest.approx([19.0, 19.0, 30.0 - 12.0])  # G -> global median 12


def test_no_features_is_a_no_op():
    X = base_frame()
    pd.testing.assert_frame_equal(FeatureEngineer(()).fit(X).transform(X), X)


def test_input_is_not_modified():
    X = base_frame()
    before = X.copy()
    FeatureEngineer(ENGINEERED_FEATURES).fit(X).transform(X)
    pd.testing.assert_frame_equal(X, before)


def test_unknown_feature_raises():
    with pytest.raises(ValueError):
        FeatureEngineer(["magic_ratio"]).fit(base_frame())


def test_unimputed_rate_raises():
    X = base_frame()
    X.loc[0, "loan_int_rate"] = np.nan
    with pytest.raises(ValueError):
        FeatureEngineer(["rate_vs_grade_median"]).fit(X)


@pytest.mark.parametrize("col,feature", [("person_income", "loan_to_income"), ("person_age", "cred_hist_to_age")])
def test_non_positive_denominator_raises(col, feature):
    X = base_frame()
    X.loc[0, col] = 0.0
    with pytest.raises(ValueError):
        FeatureEngineer([feature]).fit(X)


# ---- integration with the preprocessing pipeline ----
@pytest.fixture
def train():
    return frame(
        grades=["A", "A", "B", "B", "A", "B"],
        rates=[10.0, np.nan, 14.0, 16.0, 11.0, 15.0],
        emp=[1.0, 3.0, np.nan, 5.0, 2.0, 4.0],
    )


@pytest.mark.parametrize("family", ["tree", "linear"])
def test_pipeline_adds_engineered_columns_without_missing_values(train, family):
    pipe = build_preprocessor(family, engineered=ENGINEERED_FEATURES)
    out = pipe.fit_transform(get_xy(train)[0])
    for name in ENGINEERED_FEATURES:
        assert name in out.columns
    assert not out.isna().any().any()


def test_missing_rate_gives_zero_gap_after_imputation(train):
    pipe = build_preprocessor("tree", engineered=["rate_vs_grade_median"])
    out = pipe.fit_transform(get_xy(train)[0])
    assert out.loc[1, "rate_vs_grade_median"] == pytest.approx(0.0)  # imputed with its own grade median


def test_linear_family_scales_numeric_engineered_but_keeps_binary(train):
    pipe = build_preprocessor("linear", engineered=["loan_to_income", "high_burden"])
    out = pipe.fit_transform(get_xy(train)[0])
    assert out["loan_to_income"].mean() == pytest.approx(0.0, abs=1e-9)
    assert set(out["high_burden"].unique()) <= {0.0, 1.0}


def test_drop_removes_the_generic_column_and_rejects_others(train):
    X = get_xy(train)[0]
    out = build_preprocessor("tree", drop=["loan_percent_income"]).fit_transform(X)
    assert "loan_percent_income" not in out.columns
    with pytest.raises(ValueError):
        build_preprocessor("tree", drop=["person_income"])
    with pytest.raises(ValueError):
        build_preprocessor("tree", engineered=["magic_ratio"])


def test_default_pipeline_is_unchanged_by_the_engineering_step(train):
    X = get_xy(train)[0]
    out = build_preprocessor("tree").fit_transform(X)
    assert not set(ENGINEERED_FEATURES) & set(out.columns)
    # 7 numeric + missing flag + 2 ordinal + one-hot columns (1 per category present in this small frame)
    onehot = [c for c in out.columns if c.startswith(("person_home_ownership_", "loan_intent_"))]
    assert out.shape[1] == 7 + 1 + 2 + len(onehot)


def test_pipeline_with_engineered_features_survives_serialization(train, tmp_path):
    X = get_xy(train)[0]
    pipe = build_preprocessor("linear", engineered=ENGINEERED_FEATURES).fit(X)
    save_preprocessor(pipe, tmp_path / "p.joblib")
    loaded = load_preprocessor(tmp_path / "p.joblib")
    pd.testing.assert_frame_equal(pipe.transform(X), loaded.transform(X))


def test_selected_set_is_valid_and_replaces_the_leaky_ratio(train):
    assert set(SELECTED_ENGINEERED) <= set(ENGINEERED_FEATURES)
    assert "loan_percent_income" in SELECTED_DROP  # the provided ratio carries the leak
    out = build_selected_preprocessor("tree").fit_transform(get_xy(train)[0])
    assert "loan_to_income" in out.columns
    assert "loan_percent_income" not in out.columns
    assert not out.isna().any().any()
