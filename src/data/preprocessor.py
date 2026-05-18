"""
Data preprocessing: type casting, derived features, and persistence.

Keeps raw → processed transformation reproducible and logged.
"""

from pathlib import Path

import pandas as pd
from loguru import logger


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Types ────────────────────────────────────────────────────────────────
    df["smoker"]             = df["smoker"].astype(bool)
    df["chronic_conditions"] = df["chronic_conditions"].astype(bool)
    df["age"]                = df["age"].astype(int)

    # ── Derived features ─────────────────────────────────────────────────────
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese_I", "Obese_II+"],
        right=False,
    ).astype(str)

    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["18-29", "30-39", "40-49", "50-59", "60+"],
        right=False,
    ).astype(str)

    df["bmi_excess"] = (df["bmi"] - 25).clip(lower=0)

    import numpy as np

    # Log-exposure (offset variable for Poisson GLM)
    df["log_exposure"] = np.log(df["exposure_years"])

    # Claim-dependent features — only computed when targets are present
    if "claim_amount" in df.columns and "claim_count" in df.columns:
        df["pure_premium"] = df["claim_amount"] / df["exposure_years"]
        df["avg_claim_amount"] = df.apply(
            lambda r: r["claim_amount"] / r["claim_count"] if r["claim_count"] > 0 else 0.0,
            axis=1,
        )
    else:
        df["pure_premium"]      = 0.0
        df["avg_claim_amount"]  = 0.0

    logger.info("Preprocessing complete — added bmi_category, age_band, bmi_excess, pure_premium")
    return df


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved processed data → {path}")


def load_processed(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at {path}. Run `make data`.")
    return pd.read_parquet(path)
