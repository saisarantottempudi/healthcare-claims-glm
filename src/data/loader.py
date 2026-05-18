"""Data loading and train/test splitting."""

from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `make data` first."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    required = {
        "policy_id", "age", "gender", "bmi", "smoker", "chronic_conditions",
        "num_dependants", "region", "plan_type", "years_as_customer",
        "exposure_years", "claim_count", "claim_amount",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if (df["claim_count"] < 0).any():
        raise ValueError("claim_count contains negative values")
    if (df["claim_amount"] < 0).any():
        raise ValueError("claim_amount contains negative values")
    if (df["exposure_years"] <= 0).any():
        raise ValueError("exposure_years must be positive")
    logger.info("Schema validation passed")


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    logger.info(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train.reset_index(drop=True), test.reset_index(drop=True)
