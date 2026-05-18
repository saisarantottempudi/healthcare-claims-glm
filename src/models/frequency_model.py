"""
Poisson GLM — Claim Frequency Model
=====================================
Models the number of claims per policyholder-year.

Why Poisson?
  - claim_count is a non-negative integer (count data)
  - Variance ≈ Mean is often a reasonable first assumption
  - Log link ensures predicted rates are always positive
  - Offset term (log exposure) handles partial-year policies

Model specification:
  log E[N_i] = offset_i + β₀ + β₁·age + β₂·smoker + β₃·chronic + ...

where offset_i = log(exposure_years_i)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from sklearn.compose import ColumnTransformer

from src.features.engineer import (
    build_preprocessor,
    get_feature_names,
    prepare_design_matrix,
)


class PoissonFrequencyModel:
    """
    Poisson GLM wrapper with sklearn-style interface.

    Attributes
    ----------
    result_   : statsmodels GLMResultsWrapper after fitting
    preprocessor_ : fitted ColumnTransformer
    feature_names_ : list of feature names
    """

    def __init__(self):
        self.result_: Any = None
        self.preprocessor_: ColumnTransformer | None = None
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "PoissonFrequencyModel":
        """
        Fit the Poisson GLM on training data.

        Parameters
        ----------
        df : DataFrame with claim_count, exposure_years, and all feature columns
        """
        logger.info("Fitting Poisson GLM (frequency model) …")

        self.preprocessor_ = build_preprocessor()
        X = prepare_design_matrix(df, self.preprocessor_, fit=True)
        self.feature_names_ = get_feature_names(self.preprocessor_)

        col_names = ["const"] + self.feature_names_
        X_df = pd.DataFrame(
            sm.add_constant(X, has_constant="add"),
            columns=col_names,
        )

        y = df["claim_count"].values
        offset = np.log(df["exposure_years"].values)

        glm = sm.GLM(
            y,
            X_df,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            offset=offset,
        )
        self.result_ = glm.fit(method="irls", maxiter=100, tol=1e-8)

        logger.info(
            f"Poisson GLM fitted | AIC={self.result_.aic:.1f} | "
            f"Deviance={self.result_.deviance:.1f} | "
            f"Null deviance={self.result_.null_deviance:.1f}"
        )
        logger.info(
            f"Explained deviance: "
            f"{1 - self.result_.deviance/self.result_.null_deviance:.3%}"
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict expected claim frequency (claims per exposure year).

        Returns array of shape (n,) — the rate λ (not scaled by exposure).
        """
        if self.result_ is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
        X = prepare_design_matrix(df, self.preprocessor_, fit=False)
        col_names = ["const"] + self.feature_names_
        X_df = pd.DataFrame(
            sm.add_constant(X, has_constant="add"),
            columns=col_names,
        )
        # Offset = 0 at prediction time → returns rate per unit exposure
        pred = self.result_.predict(X_df, offset=np.zeros(len(df)))
        return pred

    def summary(self) -> str:
        if self.result_ is None:
            raise RuntimeError("Model not fitted.")
        return str(self.result_.summary())

    def coefficient_table(self) -> pd.DataFrame:
        """Returns a tidy DataFrame of coefficients, SE, z, p, CIs."""
        if self.result_ is None:
            raise RuntimeError("Model not fitted.")
        summary_df = self.result_.summary2().tables[1].copy()
        summary_df.index.name = "feature"
        summary_df.columns = ["coef", "std_err", "z", "p_value", "ci_lower", "ci_upper"]
        summary_df["exp_coef"] = np.exp(summary_df["coef"])
        return summary_df.reset_index()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"PoissonFrequencyModel saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PoissonFrequencyModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"PoissonFrequencyModel loaded ← {path}")
        return model

    @property
    def aic(self) -> float:
        return self.result_.aic if self.result_ else float("nan")

    @property
    def deviance_ratio(self) -> float:
        if self.result_ is None:
            return float("nan")
        return 1 - self.result_.deviance / self.result_.null_deviance
