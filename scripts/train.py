"""
Training pipeline — fits Poisson + Gamma GLMs and saves artifacts.

Usage:  python scripts/train.py [--config configs/config.yaml]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from loguru import logger

from src.data.loader import load_raw, validate_schema, split_data
from src.data.preprocessor import preprocess, save_processed
from src.models.frequency_model import PoissonFrequencyModel
from src.models.severity_model import GammaSeverityModel
from src.models.pure_premium import PurePremiumModel


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── 1. Data ──────────────────────────────────────────────────────────────
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

    # ── 2. Frequency model (Poisson) ─────────────────────────────────────────
    freq_model = PoissonFrequencyModel()
    freq_model.fit(train)

    logger.info("\n" + "=" * 60)
    logger.info("FREQUENCY MODEL COEFFICIENT TABLE (top 15 by |z|)")
    coef_freq = freq_model.coefficient_table()
    top15 = coef_freq.reindex(coef_freq["z"].abs().nlargest(15).index)
    logger.info("\n" + top15.to_string(index=False))

    # ── 3. Severity model (Gamma) ────────────────────────────────────────────
    sev_model = GammaSeverityModel()
    sev_model.fit(train)

    logger.info("\n" + "=" * 60)
    logger.info("SEVERITY MODEL COEFFICIENT TABLE (top 15 by |z|)")
    coef_sev = sev_model.coefficient_table()
    top15s = coef_sev.reindex(coef_sev["z"].abs().nlargest(15).index)
    logger.info("\n" + top15s.to_string(index=False))

    # ── 4. Save artifacts ────────────────────────────────────────────────────
    model_dir = ROOT / "models"
    pure_premium = PurePremiumModel(freq_model, sev_model)
    pure_premium.save(model_dir)

    logger.success("Training complete. Models saved to models/")
    logger.info(f"  Frequency AIC:      {freq_model.aic:.1f}")
    logger.info(f"  Frequency Dev ratio:{freq_model.deviance_ratio:.3%}")
    logger.info(f"  Severity  AIC:      {sev_model.aic:.1f}")
    logger.info(f"  Severity  Dev ratio:{sev_model.deviance_ratio:.3%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(ROOT / args.config)
