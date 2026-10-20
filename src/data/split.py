"""Stratified train / validation / test split.

The test set is saved to disk and must NOT be used before the final
evaluation (issue "Final evaluation on the test set").
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load import PROJECT_ROOT, TARGET

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE = 42
SPLIT_NAMES = ("train", "val", "test")


def make_splits(
    df: pd.DataFrame,
    target: str = TARGET,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stratified (train, val, test) DataFrames.

    val_size and test_size are fractions of the full dataset.
    """
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be > 0 and sum to < 1")

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=random_state
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val[target],
        random_state=random_state,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    """Write the three splits as CSV files in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in zip(SPLIT_NAMES, (train, val, test)):
        part.to_csv(out_dir / f"{name}.csv", index=False)


def load_split(name: str, in_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load one split ('train', 'val' or 'test') saved by save_splits."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"name must be one of {SPLIT_NAMES}, got {name!r}")
    path = in_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run make_splits + save_splits first.")
    return pd.read_csv(path)
