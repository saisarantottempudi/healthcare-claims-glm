"""
Notebook 01 — Exploratory Data Analysis
========================================
Business context: UK health insurer pricing review.
Auditors want evidence that the risk factors driving premiums are real.

Run as a script:  python notebooks/01_eda.py
Or convert to Jupyter:  jupytext --to notebook notebooks/01_eda.py
"""

# ── Imports ──────────────────────────────────────────────────────────────────
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
from src.data.loader import load_raw, validate_schema, split_data
from src.data.preprocessor import preprocess

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load & preprocess ─────────────────────────────────────────────────────────
df_raw = load_raw(ROOT / "data/raw/health_insurance_claims.csv")
validate_schema(df_raw)
df = preprocess(df_raw)
train, test = split_data(df)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(df.describe(include="all").T.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Claim count distribution (zero-inflated Poisson)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

cnt_vals = df["claim_count"].value_counts().sort_index()
axes[0].bar(cnt_vals.index, cnt_vals.values, color="#4C72B0", edgecolor="white", width=0.7)
axes[0].set_xlabel("Number of Claims")
axes[0].set_ylabel("Policyholders")
axes[0].set_title("Claim Count Distribution\n(zero-inflated Poisson)")
axes[0].set_xlim(-0.5, 8.5)
for x, y in zip(cnt_vals.index[:6], cnt_vals.values[:6]):
    axes[0].text(x, y + 100, f"{y:,}", ha="center", fontsize=8)

claimers = df[df["claim_count"] > 0]["claim_amount"]
axes[1].hist(claimers, bins=60, color="#DD8452", edgecolor="white", log=True)
axes[1].set_xlabel("Claim Amount (£)")
axes[1].set_ylabel("Count (log scale)")
axes[1].set_title("Claim Amount Distribution\n(claimants only — Gamma-shaped)")

plt.tight_layout()
fig.savefig(FIGURES / "01_claim_distributions.png", dpi=150)
plt.close()
print("Saved 01_claim_distributions.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Risk factor profiles
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

def freq_by_group(col):
    return df.groupby(col)["claim_count"].mean().sort_values(ascending=False)

def sev_by_group(col):
    return (
        df[df["claim_count"] > 0]
        .groupby(col)["claim_amount"]
        .mean()
        .sort_values(ascending=False)
    )

# Row 0: frequencies
for ax, col in zip(axes[0], ["region", "plan_type", "age_band"]):
    s = freq_by_group(col)
    ax.barh(s.index, s.values, color="#4C72B0")
    ax.set_xlabel("Avg Claim Count")
    ax.set_title(f"Claim Frequency by {col.replace('_',' ').title()}")

# Row 1: severities
for ax, col in zip(axes[1], ["region", "plan_type", "age_band"]):
    s = sev_by_group(col)
    ax.barh(s.index, s.values, color="#DD8452")
    ax.set_xlabel("Avg Claim Amount (£)")
    ax.set_title(f"Claim Severity by {col.replace('_',' ').title()}")

plt.tight_layout()
fig.savefig(FIGURES / "02_risk_factor_profiles.png", dpi=150)
plt.close()
print("Saved 02_risk_factor_profiles.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Smoker & chronic condition impact
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Smoker vs non-smoker claim rate
for ax, col, title in zip(
    axes,
    ["smoker", "chronic_conditions", "bmi_category"],
    ["Smoker Status", "Chronic Condition", "BMI Category"],
):
    s = df.groupby(col)["claim_count"].mean()
    ax.bar([str(k) for k in s.index], s.values, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"][:len(s)])
    ax.set_title(f"Avg Claim Count\nby {title}")
    ax.set_ylabel("Avg Claim Count")
    for i, v in enumerate(s.values):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)

plt.tight_layout()
fig.savefig(FIGURES / "03_binary_risk_factors.png", dpi=150)
plt.close()
print("Saved 03_binary_risk_factors.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Age & BMI continuous effects
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

age_bins = pd.cut(df["age"], bins=range(18, 78, 5))
freq_age = df.groupby(age_bins, observed=True)["claim_count"].mean()
axes[0].plot(range(len(freq_age)), freq_age.values, marker="o", color="#4C72B0")
axes[0].set_xticks(range(len(freq_age)))
axes[0].set_xticklabels([str(b) for b in freq_age.index], rotation=45, fontsize=8)
axes[0].set_ylabel("Avg Claim Count")
axes[0].set_title("Claim Frequency vs Age\n(monotone increasing — supports Poisson log-link)")

bmi_bins = pd.cut(df["bmi"], bins=[15, 18.5, 25, 30, 35, 55])
freq_bmi = df.groupby(bmi_bins, observed=True)["claim_count"].mean()
axes[1].plot(range(len(freq_bmi)), freq_bmi.values, marker="s", color="#DD8452")
axes[1].set_xticks(range(len(freq_bmi)))
axes[1].set_xticklabels([str(b) for b in freq_bmi.index], rotation=30, fontsize=8)
axes[1].set_ylabel("Avg Claim Count")
axes[1].set_title("Claim Frequency vs BMI\n(J-shaped — excess BMI increases risk)")

plt.tight_layout()
fig.savefig(FIGURES / "04_continuous_risk_factors.png", dpi=150)
plt.close()
print("Saved 04_continuous_risk_factors.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Correlation heatmap (numeric features)
# ─────────────────────────────────────────────────────────────────────────────
numeric_cols = ["age", "bmi", "num_dependants", "years_as_customer",
                "exposure_years", "claim_count", "claim_amount"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title("Correlation Matrix — Numeric Features")
plt.tight_layout()
fig.savefig(FIGURES / "05_correlation_matrix.png", dpi=150)
plt.close()
print("Saved 05_correlation_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Pure premium by segment (business KPI)
# ─────────────────────────────────────────────────────────────────────────────
pp = df.copy()
pp["pure_premium"] = df["claim_amount"] / df["exposure_years"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pp_region = pp.groupby("region")["pure_premium"].mean().sort_values(ascending=False)
pp_plan   = pp.groupby("plan_type")["pure_premium"].mean().sort_values(ascending=False)

axes[0].barh(pp_region.index, pp_region.values, color="#4C72B0")
axes[0].set_xlabel("Average Pure Premium (£/year)")
axes[0].set_title("Pure Premium by Region")

axes[1].barh(pp_plan.index, pp_plan.values, color="#DD8452")
axes[1].set_xlabel("Average Pure Premium (£/year)")
axes[1].set_title("Pure Premium by Plan Type")

plt.tight_layout()
fig.savefig(FIGURES / "06_pure_premium_segments.png", dpi=150)
plt.close()
print("Saved 06_pure_premium_segments.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary stats table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY BUSINESS STATS")
print("=" * 60)
print(f"Total policyholders:       {len(df):,}")
print(f"Zero-claim rate:           {(df['claim_count']==0).mean():.1%}")
print(f"Mean claims per policy:    {df['claim_count'].mean():.3f}")
print(f"Median claim amount:       £{df.loc[df['claim_count']>0,'claim_amount'].median():,.0f}")
print(f"Mean claim amount:         £{df.loc[df['claim_count']>0,'claim_amount'].mean():,.0f}")
print(f"Total claims (£M):         £{df['claim_amount'].sum()/1e6:.1f}M")
print(f"Avg pure premium (£/yr):   £{pp['pure_premium'].mean():,.0f}")
print()
print("Claims by plan type:")
print(df.groupby("plan_type")[["claim_count", "claim_amount"]].mean().round(3))
print()
print("Claims by region:")
print(df.groupby("region")["claim_count"].mean().sort_values(ascending=False).round(4))
