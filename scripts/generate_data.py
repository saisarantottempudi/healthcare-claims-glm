"""
Synthetic health insurance claims dataset generator.

Mirrors real-world actuarial data: policyholders with exposure, claim counts
drawn from Poisson, and claim amounts drawn from Gamma.  The DGP encodes
known risk factors so the GLMs can recover them.

Usage:
    python scripts/generate_data.py [--n-samples 50000] [--seed 42]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "data" / "raw" / "health_insurance_claims.csv"


def _logit(x: float) -> float:
    return np.log(x / (1 - x))


def generate_claims_dataset(n_samples: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic health insurance dataset.

    Data-generating process
    -----------------------
    True log-rate for Poisson frequency:
        log λ = -2.0
                + 0.025 * age
                + 0.30  * smoker
                + 0.40  * has_chronic
                + 0.05  * bmi_excess   (bmi - 25, clipped at 0)
                + region_effect
                + plan_effect

    True log-mean for Gamma severity (conditional on a claim occurring):
        log μ = 6.5
                + 0.018 * age
                + 0.50  * smoker
                + 0.60  * has_chronic
                + 0.04  * bmi_excess
                + region_effect
                + plan_effect
    """
    rng = np.random.default_rng(seed)

    # ── Policyholder demographics ───────────────────────────────────────────
    age = rng.integers(18, 75, size=n_samples).astype(float)
    gender = rng.choice(["Male", "Female", "Non-binary"], size=n_samples, p=[0.48, 0.49, 0.03])
    bmi = np.clip(rng.normal(26.5, 5.0, size=n_samples), 15.0, 55.0)
    smoker = rng.binomial(1, 0.18, size=n_samples)
    num_dependants = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.35, 0.25, 0.22, 0.12, 0.06])

    # Chronic condition — higher probability with age and BMI
    chronic_prob = 1 / (1 + np.exp(-(-4.0 + 0.05 * age + 0.04 * (bmi - 25))))
    has_chronic = rng.binomial(1, chronic_prob).astype(bool)

    region = rng.choice(
        ["London", "South East", "Midlands", "North England", "Scotland", "Wales"],
        size=n_samples,
        p=[0.22, 0.18, 0.20, 0.20, 0.12, 0.08],
    )
    plan_type = rng.choice(["Bronze", "Silver", "Gold", "Platinum"], size=n_samples, p=[0.30, 0.35, 0.25, 0.10])
    years_as_customer = rng.integers(1, 21, size=n_samples).astype(float)
    exposure_years = np.clip(rng.uniform(0.25, 1.0, size=n_samples), 0.25, 1.0)

    # ── Structural effects ───────────────────────────────────────────────────
    region_freq_effect = {
        "London": 0.20, "South East": 0.10, "Midlands": 0.00,
        "North England": -0.05, "Scotland": -0.10, "Wales": -0.12,
    }
    region_sev_effect = {
        "London": 0.35, "South East": 0.20, "Midlands": 0.00,
        "North England": -0.05, "Scotland": -0.10, "Wales": -0.15,
    }
    plan_freq_effect = {"Bronze": -0.20, "Silver": 0.00, "Gold": 0.15, "Platinum": 0.30}
    plan_sev_effect  = {"Bronze": -0.30, "Silver": 0.00, "Gold": 0.25, "Platinum": 0.50}

    bmi_excess = np.maximum(bmi - 25, 0)

    rf = np.array([region_freq_effect[r] for r in region])
    rs = np.array([region_sev_effect[r] for r in region])
    pf = np.array([plan_freq_effect[p] for p in plan_type])
    ps = np.array([plan_sev_effect[p]  for p in plan_type])

    # ── Poisson frequency ────────────────────────────────────────────────────
    log_lambda = (
        -2.0
        + 0.025 * age
        + 0.30  * smoker
        + 0.40  * has_chronic.astype(float)
        + 0.05  * bmi_excess
        + 0.02  * num_dependants
        + rf
        + pf
        + np.log(exposure_years)      # exposure offset
    )
    lam = np.exp(log_lambda)
    claim_count = rng.poisson(lam)

    # ── Gamma severity (per-claim amount, conditional on ≥1 claim) ──────────
    log_mu = (
        6.5
        + 0.018 * age
        + 0.50  * smoker
        + 0.60  * has_chronic.astype(float)
        + 0.04  * bmi_excess
        + 0.03  * num_dependants
        + rs
        + ps
    )
    mu_sev = np.exp(log_mu)

    # Gamma shape (dispersion ~ 1/shape); shape = 2 → moderate variability
    gamma_shape = 2.0
    gamma_scale = mu_sev / gamma_shape

    # Draw one severity per policyholder; zero it out if no claims
    raw_severity = rng.gamma(shape=gamma_shape, scale=gamma_scale)
    claim_amount = np.where(claim_count > 0, raw_severity * claim_count, 0.0).round(2)

    # ── Assemble DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "policy_id":        [f"POL{str(i).zfill(6)}" for i in range(n_samples)],
            "age":               age.astype(int),
            "gender":            gender,
            "bmi":               bmi.round(1),
            "smoker":            smoker.astype(bool),
            "chronic_conditions": has_chronic,
            "num_dependants":    num_dependants.astype(int),
            "region":            region,
            "plan_type":         plan_type,
            "years_as_customer": years_as_customer.astype(int),
            "exposure_years":    exposure_years.round(4),
            "claim_count":       claim_count.astype(int),
            "claim_amount":      claim_amount,
        }
    )

    return df


def main(n_samples: int, seed: int) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n_samples:,} synthetic policyholders (seed={seed}) …")

    df = generate_claims_dataset(n_samples=n_samples, seed=seed)

    # Quick sanity prints
    zero_claims = (df["claim_count"] == 0).mean()
    print(f"  Rows:                 {len(df):,}")
    print(f"  Zero-claim rate:      {zero_claims:.1%}")
    print(f"  Avg claim count:      {df['claim_count'].mean():.3f}")
    print(f"  Avg claim amount:     £{df.loc[df['claim_count']>0,'claim_amount'].mean():,.0f}")
    print(f"  Smoker rate:          {df['smoker'].mean():.1%}")
    print(f"  Chronic rate:         {df['chronic_conditions'].mean():.1%}")

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n_samples=args.n_samples, seed=args.seed)
