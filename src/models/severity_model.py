"""
Gamma GLM — Claim Severity Model
==================================
Models the average cost per claim, conditional on ≥1 claim occurring.

Why Gamma?
  - Claim amounts are strictly positive (GLM requires μ > 0)
  - Right-skewed with variance proportional to μ² (Gamma variance function)
  - Gamma(μ, φ): Var[Y] = φ·μ²  → coefficient of variation is constant
  - Log link ensures predicted severities are always positive
  - Far better calibrated than OLS on log(Y) because retransformation bias
    is avoided (we model Y directly, not log Y)

Model specification:
  log E[A_i | N_i > 0] = β₀ + β₁·age + β₂·smoker + β₃·chronic + ...
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


class GammaSeverityModel:
    """
    Gamma GLM wrapper with sklearn-style interface.

    Only fitted on policyholders with at least one claim.

    Attributes
    ----------
    result_       : statsmodels GLMResultsWrapper after fitting
    preprocessor_ : fitted ColumnTransformer
    feature_names_ : list of feature names
    dispersion_   : estimated Gamma dispersion parameter φ
    """

    def __init__(self):
        self.result_: Any = None
        self.preprocessor_: ColumnTransformer | None = None
        self.feature_names_: list[str] = []
        self.dispersion_: float = float("nan")

    def fit(self, df: pd.DataFrame) -> "GammaSeverityModel":
        """
        Fit Gamma GLM on claimants only.

        Parameters
        ----------
        df : Full training DataFrame; rows with claim_count == 0 are dropped.
        """
        claimants = df[df["claim_count"] > 0].reset_index(drop=True)
        logger.info(
            f"Fitting Gamma GLM on {len(claimants):,} claimants "
            f"({len(claimants)/len(df):.1%} of training set) …"
        )

        self.preprocessor_ = build_preprocessor()
        X = prepare_design_matrix(claimants, self.preprocessor_, fit=True)
        self.feature_names_ = get_feature_names(self.preprocessor_)

        col_names = ["const"] + self.feature_names_
        X_df = pd.DataFrame(
            sm.add_constant(X, has_constant="add"),
            columns=col_names,
        )

        # Target: average claim amount per claim
        y = (claimants["claim_amount"] / claimants["claim_count"]).values

        glm = sm.GLM(
            y,
            X_df,
            family=sm.families.Gamma(link=sm.families.links.Log()),
        )
        self.result_ = glm.fit(method="irls", maxiter=100, tol=1e-8)
        self.dispersion_ = self.result_.scale   # φ̂

        logger.info(
            f"Gamma GLM fitted | AIC={self.result_.aic:.1f} | "
            f"Deviance={self.result_.deviance:.1f} | "
            f"φ̂={self.dispersion_:.4f}"
        )
        logger.info(
            f"Explained deviance: "
            f"{1 - self.result_.deviance/self.result_.null_deviance:.3%}"
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict expected severity (£ per claim) for any policyholder.

        Rows with claim_count == 0 will receive a predicted severity
        (representing what a claim *would* cost if one occurred).
        """
        if self.result_ is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
        X = prepare_design_matrix(df, self.preprocessor_, fit=False)
        col_names = ["const"] + self.feature_names_
        X_df = pd.DataFrame(
            sm.add_constant(X, has_constant="add"),
            columns=col_names,
        )
        pred = self.result_.predict(X_df)
        return pred

    def summary(self) -> str:
        if self.result_ is None:
            raise RuntimeError("Model not fitted.")
        return str(self.result_.summary())

    def coefficient_table(self) -> pd.DataFrame:
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
        logger.info(f"GammaSeverityModel saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GammaSeverityModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"GammaSeverityModel loaded ← {path}")
        return model

    @property
    def aic(self) -> float:
        return self.result_.aic if self.result_ else float("nan")

    @property
    def deviance_ratio(self) -> float:
        if self.result_ is None:
            return float("nan")
        return 1 - self.result_.deviance / self.result_.null_deviance
