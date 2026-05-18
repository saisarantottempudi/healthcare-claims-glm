"""
Evaluation pipeline — loads models, runs test-set diagnostics, saves figures.

Usage:  python scripts/evaluate.py [--config configs/config.yaml]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from loguru import logger

from src.data.preprocessor import load_processed
from src.evaluation.metrics import (
    calibration_by_decile,
    gini_coefficient,
    gamma_deviance,
    lorenz_curve,
    mae,
    mape,
    pearson_residuals,
    poisson_deviance,
    rmse,
)
from src.models.pure_premium import PurePremiumModel

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Load test data and models ────────────────────────────────────────────
    test = load_processed(ROOT / cfg["data"]["test_path"])
    model = PurePremiumModel.load(ROOT / "models")

    preds = model.predict(test)
    test  = pd.concat([test.reset_index(drop=True), preds], axis=1)

    # Frequency evaluation (all rows)
    y_freq  = test["claim_count"].values
    y_freq_hat = test["pred_frequency"].values

    # Severity evaluation (claimants only)
    claimants = test[test["claim_count"] > 0].copy()
    y_sev     = (claimants["claim_amount"] / claimants["claim_count"]).values
    y_sev_hat = claimants["pred_severity"].values

    # Pure premium
    y_pp     = test["pure_premium"].values
    y_pp_hat = test["pred_pure_premium"].values

    # ── Print metrics ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("FREQUENCY MODEL (Poisson GLM) — Test Set Metrics")
    logger.info("=" * 65)
    logger.info(f"  Poisson Deviance (unit):  {poisson_deviance(y_freq, y_freq_hat):.4f}")
    logger.info(f"  MAE:                      {mae(y_freq, y_freq_hat):.4f}")
    logger.info(f"  RMSE:                     {rmse(y_freq, y_freq_hat):.4f}")
    logger.info(f"  Gini (lift):              {gini_coefficient(y_freq, y_freq_hat):.4f}")

    logger.info("\n" + "=" * 65)
    logger.info("SEVERITY MODEL (Gamma GLM) — Test Set Metrics (claimants only)")
    logger.info("=" * 65)
    logger.info(f"  Gamma Deviance (unit):    {gamma_deviance(y_sev, y_sev_hat):.4f}")
    logger.info(f"  MAE:                      £{mae(y_sev, y_sev_hat):,.0f}")
    logger.info(f"  RMSE:                     £{rmse(y_sev, y_sev_hat):,.0f}")
    logger.info(f"  MAPE:                     {mape(y_sev, y_sev_hat):.2%}")
    logger.info(f"  Gini (lift):              {gini_coefficient(y_sev, y_sev_hat):.4f}")

    logger.info("\n" + "=" * 65)
    logger.info("PURE PREMIUM — Test Set Metrics")
    logger.info("=" * 65)
    logger.info(f"  MAE:                      £{mae(y_pp, y_pp_hat):,.0f}")
    logger.info(f"  RMSE:                     £{rmse(y_pp, y_pp_hat):,.0f}")
    logger.info(f"  Gini (lift):              {gini_coefficient(y_pp, y_pp_hat):.4f}")

    # ── FIG 7 — Lorenz curves ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, y_t, y_s, title in [
        (axes[0], y_freq, y_freq_hat, "Claim Frequency\n(Poisson GLM)"),
        (axes[1], y_pp,  y_pp_hat,   "Pure Premium\n(Frequency × Severity)"),
    ]:
        x, y = lorenz_curve(y_t, y_s)
        ax.plot(x, y, label=f"GLM  (Gini={gini_coefficient(y_t, y_s):.3f})", lw=2, color="#4C72B0")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
        ax.fill_between(x, y, x, alpha=0.15, color="#4C72B0")
        ax.set_xlabel("Cumulative share of policyholders\n(ordered by predicted risk)")
        ax.set_ylabel("Cumulative share of claims")
        ax.set_title(f"Lorenz Curve — {title}")
        ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES / "07_lorenz_curves.png", dpi=150)
    plt.close()
    logger.info("Saved 07_lorenz_curves.png")

    # ── FIG 8 — Calibration plots ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, y_t, y_s, title, unit in [
        (axes[0], y_freq, y_freq_hat, "Frequency (Poisson)", "claims/yr"),
        (axes[1], y_pp,  y_pp_hat,   "Pure Premium",         "£/yr"),
    ]:
        cal = calibration_by_decile(y_t, y_s)
        ax.plot(cal["pred_mean"], cal["actual_mean"], "o-", color="#4C72B0", lw=2, label="Actual")
        lim = max(cal["pred_mean"].max(), cal["actual_mean"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="Perfect calibration")
        ax.set_xlabel(f"Predicted ({unit})")
        ax.set_ylabel(f"Actual ({unit})")
        ax.set_title(f"Calibration by Decile\n{title}")
        ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES / "08_calibration.png", dpi=150)
    plt.close()
    logger.info("Saved 08_calibration.png")

    # ── FIG 9 — Residual diagnostics ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Pearson residuals vs fitted (frequency)
    pr_freq = pearson_residuals(y_freq, y_freq_hat, family="poisson")
    axes[0].scatter(y_freq_hat, pr_freq, alpha=0.1, s=5, color="#4C72B0")
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_xlabel("Fitted values (λ)")
    axes[0].set_ylabel("Pearson residual")
    axes[0].set_title("Frequency — Residuals vs Fitted")
    axes[0].set_ylim(-5, 10)

    # Pearson residuals vs fitted (severity)
    pr_sev = pearson_residuals(y_sev, y_sev_hat, family="gamma")
    axes[1].scatter(y_sev_hat, pr_sev, alpha=0.15, s=5, color="#DD8452")
    axes[1].axhline(0, color="red", lw=1)
    axes[1].set_xlabel("Fitted values (μ_sev)")
    axes[1].set_ylabel("Pearson residual")
    axes[1].set_title("Severity — Residuals vs Fitted")
    axes[1].set_ylim(-5, 10)

    # Coefficient importance (frequency model — top 15)
    coef_f = model.frequency_model.coefficient_table()
    coef_f = coef_f[coef_f["feature"] != "const"].nlargest(15, "z")
    coef_f["color"] = coef_f["coef"].apply(lambda v: "#DD8452" if v > 0 else "#4C72B0")
    axes[2].barh(
        coef_f["feature"].str.replace("num__|cat__|bin__", "", regex=True),
        coef_f["exp_coef"] - 1,
        color=coef_f["color"],
    )
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].set_xlabel("exp(β) − 1  (multiplicative effect on rate)")
    axes[2].set_title("Frequency GLM — Top Risk Factors")

    plt.tight_layout()
    fig.savefig(FIGURES / "09_diagnostics.png", dpi=150)
    plt.close()
    logger.info("Saved 09_diagnostics.png")

    # ── FIG 10 — Coefficient plot (forest plot style) ────────────────────────
    for (model_obj, label, fname) in [
        (model.frequency_model, "Poisson GLM — Frequency", "10a_coef_frequency"),
        (model.severity_model,  "Gamma GLM — Severity",    "10b_coef_severity"),
    ]:
        ct = model_obj.coefficient_table()
        ct = ct[ct["feature"] != "const"].sort_values("coef", ascending=True)
        ct["feature_clean"] = ct["feature"].str.replace(
            r"num__|cat__|bin__", "", regex=True
        )

        fig, ax = plt.subplots(figsize=(8, max(6, len(ct) * 0.35)))
        colors = ["#DD8452" if v > 0 else "#4C72B0" for v in ct["coef"]]
        ax.barh(ct["feature_clean"], ct["coef"], color=colors, height=0.6)
        ax.errorbar(
            ct["coef"], ct["feature_clean"],
            xerr=[ct["coef"] - ct["ci_lower"], ct["ci_upper"] - ct["coef"]],
            fmt="none", color="black", capsize=3, lw=1,
        )
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Coefficient (log-scale)")
        ax.set_title(f"Coefficient Plot\n{label}")
        plt.tight_layout()
        fig.savefig(FIGURES / f"{fname}.png", dpi=150)
        plt.close()
        logger.info(f"Saved {fname}.png")

    logger.success("Evaluation complete. All figures saved to reports/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(ROOT / args.config)
