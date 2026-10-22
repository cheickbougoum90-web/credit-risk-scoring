import numpy as np
import pandas as pd

from src.data.clean import (
    AGE_MAX,
    EMP_LENGTH_MAX,
    impossible_mask,
    remove_impossible_values,
)


def make_df():
    return pd.DataFrame(
        {
            "person_age": [25, 144, 30, AGE_MAX, 123, 40],
            "person_emp_length": [3.0, 2.0, 123.0, EMP_LENGTH_MAX, np.nan, np.nan],
            "loan_status": [0, 1, 0, 1, 0, 1],
        }
    )


def test_flags_impossible_age_and_employment_length():
    mask = impossible_mask(make_df())
    assert mask.tolist() == [False, True, True, False, True, False]


def test_boundaries_are_kept():
    df = make_df()
    kept = remove_impossible_values(df)
    assert AGE_MAX in kept["person_age"].tolist()
    assert EMP_LENGTH_MAX in kept["person_emp_length"].tolist()


def test_missing_employment_length_is_not_removed():
    df = make_df()
    kept = remove_impossible_values(df)
    assert kept["person_emp_length"].isna().sum() == 1  # the (40, NaN) row survives


def test_index_is_reset_and_input_is_not_modified():
    df = make_df()
    before = df.copy()
    kept = remove_impossible_values(df)
    assert kept.index.tolist() == list(range(len(kept)))
    pd.testing.assert_frame_equal(df, before)


def test_clean_frame_has_no_impossible_rows():
    kept = remove_impossible_values(make_df())
    assert not impossible_mask(kept).any()
