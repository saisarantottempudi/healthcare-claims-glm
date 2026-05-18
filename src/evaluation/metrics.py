"""
Model evaluation metrics for GLMs.

Metrics used by actuaries and MNC data science teams:
  - Deviance (Poisson / Gamma)
  - Mean Absolute Error, RMSE
  - Gini coefficient / Lorenz curve (lift analysis)
  - Pearson residuals
  - Calibration (predicted vs actual by decile)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ── Deviance metrics ──────────────────────────────────────────────────────────

def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unit Poisson deviance (per observation)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Avoid log(0): where y_true == 0, the term is 2*(y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y_true > 0, y_true / y_pred, 1.0)
        log_ratio = np.where(y_true > 0, np.log(ratio), 0.0)
    d = 2 * (y_true * log_ratio - (y_true - y_pred))
    return float(d.mean())


def gamma_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unit Gamma deviance (per observation)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = y_true / y_pred
        d = -2 * (np.log(ratio) - (ratio - 1))
    return float(d.mean())


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    return float(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean())


# ── Lorenz / Gini ─────────────────────────────────────────────────────────────

def lorenz_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (x, y) coordinates of the Lorenz curve.

    Sort policies from lowest to highest predicted risk (y_score),
    then plot cumulative share of policies vs cumulative share of claims.
    """
    order = np.argsort(y_score)
    y_sorted = np.asarray(y_true, dtype=float)[order]
    x = np.linspace(0, 1, len(y_sorted))
    y = np.cumsum(y_sorted) / y_sorted.sum()
    return x, y


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Gini concentration coefficient.  Ranges 0 (random) to 1 (perfect lift).

    When policies are sorted ascending by predicted risk, the Lorenz curve
    bows below the 45° diagonal.  Gini = 1 − 2·AUC_lorenz.
    """
    x, y = lorenz_curve(y_true, y_score)
    auc = np.trapezoid(y, x)
    return float(1 - 2 * auc)


# ── Calibration ───────────────────────────────────────────────────────────────

def calibration_by_decile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compare predicted vs actual by decile of predicted score.

    Standard actuarial goodness-of-fit diagnostic.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["decile"] = pd.qcut(df["y_pred"], q=n_bins, labels=False, duplicates="drop")
    result = (
        df.groupby("decile")
        .agg(
            n=("y_true", "count"),
            actual_mean=("y_true", "mean"),
            pred_mean=("y_pred", "mean"),
        )
        .reset_index()
    )
    result["ratio"] = result["actual_mean"] / result["pred_mean"]
    return result


# ── Residual diagnostics ──────────────────────────────────────────────────────

def pearson_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, family: str = "poisson"
) -> np.ndarray:
    """Pearson residuals: (y - μ) / sqrt(V(μ))"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if family == "poisson":
        var = y_pred
    elif family == "gamma":
        var = y_pred ** 2
    else:
        raise ValueError(f"Unknown family: {family}")
    return (y_true - y_pred) / np.sqrt(var)


def deviance_residuals_poisson(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Signed deviance residuals for Poisson."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(y_true > 0, np.log(y_true / y_pred), 0.0)
        di = 2 * (y_true * log_ratio - (y_true - y_pred))
    sign = np.sign(y_true - y_pred)
    return sign * np.sqrt(np.abs(di))
