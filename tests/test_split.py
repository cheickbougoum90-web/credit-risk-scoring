import numpy as np
import pandas as pd
import pytest

from src.data.split import load_split, make_splits, save_splits


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame(
        {
            "row_id": np.arange(n),
            "x": rng.normal(size=n),
            "loan_status": (rng.random(n) < 0.22).astype(int),
        }
    )


def test_sizes_are_60_20_20(df):
    train, val, test = make_splits(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert abs(len(train) - 600) <= 2
    assert abs(len(val) - 200) <= 2
    assert abs(len(test) - 200) <= 2


def test_no_overlap_between_splits(df):
    train, val, test = make_splits(df)
    ids = [set(p["row_id"]) for p in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_target_is_stratified(df):
    overall = df["loan_status"].mean()
    for part in make_splits(df):
        assert abs(part["loan_status"].mean() - overall) < 0.01


def test_split_is_reproducible(df):
    a = make_splits(df)
    b = make_splits(df)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)


def test_invalid_sizes_raise(df):
    with pytest.raises(ValueError):
        make_splits(df, val_size=0.6, test_size=0.5)


def test_save_and_load_roundtrip(df, tmp_path):
    train, val, test = make_splits(df)
    save_splits(train, val, test, out_dir=tmp_path)
    pd.testing.assert_frame_equal(load_split("train", in_dir=tmp_path), train)
    with pytest.raises(ValueError):
        load_split("holdout", in_dir=tmp_path)
