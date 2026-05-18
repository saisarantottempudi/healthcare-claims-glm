"""
Feature engineering pipeline compatible with scikit-learn.

Produces a numpy design matrix from a raw DataFrame — ready to be
passed directly to statsmodels GLM or sklearn estimators.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "bmi_excess",
    "num_dependants",
    "years_as_customer",
]

CATEGORICAL_FEATURES = [
    "gender",
    "region",
    "plan_type",
    "bmi_category",
    "age_band",
]

BINARY_FEATURES = [
    "smoker",
    "chronic_conditions",
]


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Converts boolean / 0-1 columns to float."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].astype(float)
        return X


class InteractionEncoder(BaseEstimator, TransformerMixin):
    """Add domain-informed interaction terms before the main transformer."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Smoker × chronic is clinically meaningful and additive on log-scale
        X["smoker_x_chronic"] = X["smoker"].astype(float) * X["chronic_conditions"].astype(float)
        # Age × smoking risk amplifier
        X["age_x_smoker"] = X["age"].astype(float) * X["smoker"].astype(float) / 100
        return X


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a fitted-ready ColumnTransformer.

    Numeric:     StandardScaler
    Categorical: OneHotEncoder (drop='first' for GLM identifiability)
    Binary:      passthrough (already 0/1)
    Interaction: passthrough
    """
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_pipeline,    NUMERIC_FEATURES),
            ("cat",  categorical_pipeline, CATEGORICAL_FEATURES),
            ("bin",  "passthrough",        BINARY_FEATURES + ["smoker_x_chronic", "age_x_smoker"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names after fitting."""
    return list(preprocessor.get_feature_names_out())


def prepare_design_matrix(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    fit: bool = False,
) -> np.ndarray:
    """Add interaction columns and run through the preprocessor."""
    interaction = InteractionEncoder()
    df_aug = interaction.transform(df)
    if fit:
        X = preprocessor.fit_transform(df_aug)
    else:
        X = preprocessor.transform(df_aug)
    return np.asarray(X, dtype=np.float64)
