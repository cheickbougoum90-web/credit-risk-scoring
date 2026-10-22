"""Deterministic cleaning rules decided in the EDA (issue #5).

Rows with impossible values are removed with the SAME rule on the train,
validation and test sets. The rule is fixed (not learned from data), so it
cannot leak information between splits.

Imputation of missing values is NOT done here: it is learned on the training
set and therefore lives in the scikit-learn pipeline (issue "Build the
preprocessing pipeline").
"""
import pandas as pd

from src.data.split import load_split

# Bounds shared with the API input validation.
AGE_MAX = 100
EMP_LENGTH_MAX = 60


def impossible_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows with an impossible age or employment length.

    Missing employment lengths are NOT flagged (they are imputed later).
    """
    return (df["person_age"] > AGE_MAX) | (df["person_emp_length"] > EMP_LENGTH_MAX)


def remove_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df without impossible rows, with a fresh index."""
    return df.loc[~impossible_mask(df)].reset_index(drop=True)


def load_clean_split(name: str) -> pd.DataFrame:
    """Load a saved split ('train', 'val' or 'test') and apply the cleaning rule."""
    return remove_impossible_values(load_split(name))
