from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv"
TARGET = "loan_status"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw credit risk dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset from Kaggle "
            "(laotse/credit-risk-dataset) and place it in data/raw/."
        )
    return pd.read_csv(path)
