"""
MLflow-instrumented training run.

Logs parameters, metrics, and model artifacts to the local MLflow store.
In production this would point at a remote tracking server (Databricks, AWS).

Usage:  python scripts/train_mlflow.py [--config configs/config.yaml]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow
import yaml
from loguru import logger

from src.data.loader import load_raw, validate_schema, split_data
from src.data.preprocessor import preprocess, save_processed
from src.evaluation.metrics import (
    calibration_by_decile,
    gini_coefficient,
    gamma_deviance,
    mae,
    poisson_deviance,
    rmse,
)
from src.models.frequency_model import PoissonFrequencyModel
from src.models.severity_model import GammaSeverityModel
from src.models.pure_premium import PurePremiumModel


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(ROOT / cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="poisson_gamma_glm") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # ── Log config params ────────────────────────────────────────────────
        mlflow.log_params({
            "n_samples":    cfg["data"]["n_samples"],
            "test_size":    cfg["data"]["test_size"],
            "random_state": cfg["data"]["random_state"],
            "freq_family":  cfg["frequency_model"]["family"],
            "freq_link":    cfg["frequency_model"]["link"],
            "sev_family":   cfg["severity_model"]["family"],
            "sev_link":     cfg["severity_model"]["link"],
        })

        # ── Data ─────────────────────────────────────────────────────────────
        df_raw = load_raw(ROOT / cfg["data"]["raw_path"])
        validate_schema(df_raw)
        df = preprocess(df_raw)
        train, test = split_data(
            df,
            test_size=cfg["data"]["test_size"],
            random_state=cfg["data"]["random_state"],
        )
        save_processed(train, ROOT / cfg["data"]["train_path"])
        save_processed(test,  ROOT / cfg["data"]["test_path"])

        mlflow.log_params({
            "train_rows": len(train),
            "test_rows":  len(test),
            "claimant_rate": f"{(train['claim_count']>0).mean():.3f}",
        })

        # ── Train ────────────────────────────────────────────────────────────
        freq_model = PoissonFrequencyModel()
        freq_model.fit(train)

        sev_model = GammaSeverityModel()
        sev_model.fit(train)

        # ── Evaluate on test set ─────────────────────────────────────────────
        pure_premium = PurePremiumModel(freq_model, sev_model)
        preds = pure_premium.predict(test)

        y_freq = test["claim_count"].values
        y_freq_hat = preds["pred_frequency"].values

        claimants = test[test["claim_count"] > 0].copy()
        y_sev     = (claimants["claim_amount"] / claimants["claim_count"]).values
        y_sev_hat = preds.loc[claimants.index, "pred_severity"].values

        y_pp     = test["pure_premium"].values
        y_pp_hat = preds["pred_pure_premium"].values

        # Log model metrics
        mlflow.log_metrics({
            # Frequency
            "freq_aic":              freq_model.aic,
            "freq_deviance_ratio":   freq_model.deviance_ratio,
            "freq_poisson_deviance": poisson_deviance(y_freq, y_freq_hat),
            "freq_mae":              mae(y_freq, y_freq_hat),
            "freq_rmse":             rmse(y_freq, y_freq_hat),
            "freq_gini":             gini_coefficient(y_freq, y_freq_hat),
            # Severity
            "sev_aic":               sev_model.aic,
            "sev_deviance_ratio":    sev_model.deviance_ratio,
            "sev_gamma_deviance":    gamma_deviance(y_sev, y_sev_hat),
            "sev_mae":               mae(y_sev, y_sev_hat),
            "sev_rmse":              rmse(y_sev, y_sev_hat),
            "sev_gini":              gini_coefficient(y_sev, y_sev_hat),
            # Pure premium
            "pp_mae":                mae(y_pp, y_pp_hat),
            "pp_rmse":               rmse(y_pp, y_pp_hat),
            "pp_gini":               gini_coefficient(y_pp, y_pp_hat),
        })

        # Log calibration table as artifact
        cal = calibration_by_decile(y_freq, y_freq_hat)
        cal_path = ROOT / "reports" / "calibration_table.csv"
        cal.to_csv(cal_path, index=False)
        mlflow.log_artifact(str(cal_path))

        # Log coefficient tables
        freq_coef_path = ROOT / "reports" / "freq_coefficients.csv"
        sev_coef_path  = ROOT / "reports" / "sev_coefficients.csv"
        freq_model.coefficient_table().to_csv(freq_coef_path, index=False)
        sev_model.coefficient_table().to_csv(sev_coef_path, index=False)
        mlflow.log_artifact(str(freq_coef_path))
        mlflow.log_artifact(str(sev_coef_path))

        # Log figures
        import os
        for fig_path in (ROOT / "reports" / "figures").glob("*.png"):
            mlflow.log_artifact(str(fig_path), "figures")

        # ── Save model artifacts ─────────────────────────────────────────────
        model_dir = ROOT / "models"
        pure_premium.save(model_dir)
        mlflow.log_artifact(str(model_dir / "frequency_model.pkl"), "models")
        mlflow.log_artifact(str(model_dir / "severity_model.pkl"),  "models")

        logger.success(
            f"MLflow run complete. View at: mlflow ui --backend-store-uri {ROOT / cfg['mlflow']['tracking_uri']}"
        )
        logger.info(f"  Run ID: {run.info.run_id}")
        logger.info(f"  Pure Premium Gini: {gini_coefficient(y_pp, y_pp_hat):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(ROOT / args.config)
