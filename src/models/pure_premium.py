"""
Pure Premium Model — Frequency × Severity
==========================================
Combines the Poisson frequency model and Gamma severity model to produce
a single expected annual cost estimate per policyholder.

    Pure Premium_i = E[N_i | x_i] × E[A_i | N_i > 0, x_i]

This two-part decomposition is the standard actuarial approach because:
  1. Zero-inflation: most policyholders have zero claims (different process)
  2. Separate risk drivers: frequency and severity may respond differently
     to the same covariate (e.g. Platinum plan massively raises severity but
     also mildly raises frequency)
  3. Regulatory interpretability: actuaries must justify each component
     independently when filing rates with regulators
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.frequency_model import PoissonFrequencyModel
from src.models.severity_model import GammaSeverityModel


class PurePremiumModel:
    """
    Composite model: Pure Premium = Frequency × Severity.

    Parameters
    ----------
    frequency_model : fitted PoissonFrequencyModel
    severity_model  : fitted GammaSeverityModel
    """

    def __init__(
        self,
        frequency_model: PoissonFrequencyModel,
        severity_model: GammaSeverityModel,
    ):
        self.frequency_model = frequency_model
        self.severity_model  = severity_model

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with frequency, severity, and pure premium predictions.

        Parameters
        ----------
        df : preprocessed DataFrame (output of src.data.preprocessor.preprocess)

        Returns
        -------
        DataFrame with columns:
            pred_frequency  — expected annual claim count
            pred_severity   — expected cost per claim (£)
            pred_pure_premium — expected annual claim cost (£)
        """
        freq = self.frequency_model.predict(df)       # rate per year
        sev  = self.severity_model.predict(df)         # £ per claim

        return pd.DataFrame(
            {
                "pred_frequency":    freq,
                "pred_severity":     sev,
                "pred_pure_premium": freq * sev,
            },
            index=df.index,
        )

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.frequency_model.save(directory / "frequency_model.pkl")
        self.severity_model.save(directory  / "severity_model.pkl")
        logger.info(f"PurePremiumModel artifacts saved → {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "PurePremiumModel":
        directory = Path(directory)
        freq_model = PoissonFrequencyModel.load(directory / "frequency_model.pkl")
        sev_model  = GammaSeverityModel.load(directory  / "severity_model.pkl")
        logger.info(f"PurePremiumModel loaded ← {directory}")
        return cls(freq_model, sev_model)
