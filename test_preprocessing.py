import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import (
    FEATURES,
    GroupMedianImputer,
    build_preprocessor,
    get_xy,
    load_preprocessor,
    save_preprocessor,
)


def frame(grades, rates, emp, ages=None, homes=None):
    n = len(grades)
    return pd.DataFrame(
        {
            "person_age": ages if ages is not None else [25 + 5 * i for i in range(n)],
            "person_income": [30000.0 + 10000 * i for i in range(n)],
            "person_home_ownership": homes if homes is not None else ["RENT"] * n,
            "person_emp_length": emp,
            "loan_intent": ["EDUCATION"] * n,
            "loan_grade": grades,
            "loan_amnt": [5000.0 + 1000 * i for i in range(n)],
            "loan_int_rate": rates,
            "loan_percent_income": [0.1 + 0.02 * i for i in range(n)],
            "cb_person_default_on_file": ["N", "Y"] * (n // 2) + ["N"] * (n % 2),
            "cb_person_cred_hist_length": [2 + i for i in range(n)],
            "loan_status": [0, 1] * (n // 2) + [0] * (n % 2),
        }
    )


@pytest.fixture
def train():
    # grade A median rate = 10.0 ; grade B median = 15.0 ; global median = 14.0
    # employment length median = 3.0
    return frame(
        grades=["A", "A", "B", "B"],
        rates=[10.0, np.nan, 14.0, 16.0],
        emp=[1.0, 3.0, np.nan, 5.0],
    )


@pytest.fixture
def other():
    return frame(
        grades=["A", "B", "G"],
        rates=[np.nan, np.nan, np.nan],
        emp=[np.nan, 100.0, np.nan],
    )


def test_group_median_imputer_uses_training_medians_only(train, other):
    imp = GroupMedianImputer().fit(train)
    out = imp.transform(other)
    assert out.loc[0, "loan_int_rate"] == 10.0  # grade A, learned on train
    assert out.loc[1, "loan_int_rate"] == 15.0  # grade B, learned on train


def test_unseen_grade_falls_back_to_global_median(train, other):
    out = GroupMedianImputer().fit(train).transform(other)
    assert out.loc[2, "loan_int_rate"] == 14.0  # grade G never seen in train


def test_observed_values_are_not_modified(train):
    out = GroupMedianImputer().fit(train).transform(train)
    assert out.loc[0, "loan_int_rate"] == 10.0
    assert out.loc[3, "loan_int_rate"] == 16.0


def test_imputer_does_not_modify_input(train):
    before = train.copy()
    GroupMedianImputer().fit(train).transform(train)
    pd.testing.assert_frame_equal(train, before)


def test_imputer_requires_observed_values():
    df = frame(["A", "B"], [np.nan, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError):
        GroupMedianImputer().fit(df)


def test_employment_length_median_and_indicator_come_from_train(train, other):
    pipe = build_preprocessor("tree")
    pipe.fit(get_xy(train)[0])
    out = pipe.transform(get_xy(other)[0])
    assert out["person_emp_length"].tolist() == [3.0, 100.0, 3.0]  # train median, not 100
    assert out["person_emp_length_missing"].tolist() == [1.0, 0.0, 1.0]


@pytest.mark.parametrize("family", ["tree", "linear"])
def test_no_missing_values_after_transform(train, other, family):
    pipe = build_preprocessor(family)
    pipe.fit(get_xy(train)[0])
    assert not pipe.transform(get_xy(train)[0]).isna().any().any()
    assert not pipe.transform(get_xy(other)[0]).isna().any().any()


def test_ordinal_encoding_of_grade_and_default_flag(train):
    pipe = build_preprocessor("tree")
    X = get_xy(train)[0]
    out = pipe.fit_transform(X)
    assert out["loan_grade"].tolist() == [0.0, 0.0, 1.0, 1.0]  # A=0, B=1
    assert out["cb_person_default_on_file"].tolist() == [0.0, 1.0, 0.0, 1.0]  # N=0, Y=1


def test_unknown_grade_raises(train):
    pipe = build_preprocessor("tree").fit(get_xy(train)[0])
    bad = get_xy(frame(["Z"], [10.0], [2.0]))[0]
    with pytest.raises(ValueError):
        pipe.transform(bad)


def test_unknown_nominal_category_is_ignored(train):
    pipe = build_preprocessor("tree").fit(get_xy(train)[0])
    new = get_xy(frame(["A"], [10.0], [2.0], homes=["WEIRD"]))[0]
    out = pipe.transform(new)
    home_cols = [c for c in out.columns if c.startswith("person_home_ownership_")]
    assert home_cols and (out[home_cols].to_numpy() == 0).all()


def test_tree_family_keeps_raw_scales_linear_family_scales(train):
    X = get_xy(train)[0]
    tree = build_preprocessor("tree").fit_transform(X)
    linear = build_preprocessor("linear").fit_transform(X)
    assert tree["person_income"].tolist() == X["person_income"].tolist()
    assert linear["person_age"].mean() == pytest.approx(0.0, abs=1e-9)
    assert linear["person_age"].std(ddof=0) == pytest.approx(1.0)
    assert linear["person_income"].std(ddof=0) == pytest.approx(1.0)


def test_feature_names_are_consistent(train):
    pipe = build_preprocessor("tree")
    out = pipe.fit_transform(get_xy(train)[0])
    assert list(pipe.get_feature_names_out()) == list(out.columns)
    assert "person_emp_length_missing" in out.columns
    assert "loan_int_rate" in out.columns


def test_missing_required_column_raises(train):
    X = get_xy(train)[0].drop(columns=["person_age"])
    with pytest.raises(ValueError):
        build_preprocessor("tree").fit(X)


def test_invalid_family_raises():
    with pytest.raises(ValueError):
        build_preprocessor("neural")


def test_get_xy_returns_fixed_column_order(train):
    X, y = get_xy(train)
    assert list(X.columns) == FEATURES
    assert y.tolist() == train["loan_status"].tolist()


def test_serialization_roundtrip_gives_same_output(train, other, tmp_path):
    pipe = build_preprocessor("linear").fit(get_xy(train)[0])
    path = tmp_path / "models" / "preprocessor.joblib"
    save_preprocessor(pipe, path)
    loaded = load_preprocessor(path)
    pd.testing.assert_frame_equal(
        pipe.transform(get_xy(other)[0]), loaded.transform(get_xy(other)[0])
    )
