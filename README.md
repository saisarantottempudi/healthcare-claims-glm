# Healthcare Claims GLM

**End-to-end Generalised Linear Model (GLM) system for health insurance claims pricing.**

A production-grade data science project modelling claim **frequency** (Poisson GLM) and claim **severity** (Gamma GLM) for a UK health insurer — the standard actuarial two-part model used at AXA, Aviva, BUPA, Cigna and similar companies.

---

## Business Problem

A health insurer needs to price policies fairly and profitably. Charging everyone the same premium leads to adverse selection — healthy low-risk customers leave and unhealthy high-risk customers stay, destroying the risk pool.

**Solution**: Build a **pure premium model** that estimates each policyholder's expected annual claims cost:

```
Pure Premium = E[Frequency] × E[Severity]
             = E[# claims / year | covariates]
             × E[£ per claim | covariates, claim occurred]
```

This two-part decomposition is required because:
- Zero-inflated counts (67% of policyholders make zero claims in a year)
- Frequency and severity respond **differently** to the same covariates
- Regulators require each component to be justified independently

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                            │
│  generate_data.py  →  50k synthetic policyholders (Poisson + Gamma)   │
│  src/data/loader.py, preprocessor.py                                  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│  FEATURE ENGINEERING                                                   │
│  StandardScaler (numeric) + OneHotEncoder (categorical)                │
│  + interaction terms: smoker×chronic, age×smoker                       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│  MODELS                                                                │
│  PoissonFrequencyModel   (log λ = Xβ + log_exposure)                  │
│  GammaSeverityModel      (log μ = Xβ,  on claimants only)             │
│  PurePremiumModel        = frequency_model × severity_model            │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌────────────────┬──────────────────▼──────────────────────┐
│  TRACKING      │  SERVING                                 │
│  MLflow        │  FastAPI  ←── POST /predict              │
│  (params +     │             POST /predict/batch          │
│   metrics +    │             GET  /health                 │
│   artifacts)   │             GET  /model/info             │
└────────────────┴──────────────────────────────────────────┘
```

---

## Model Results

### Poisson GLM (Frequency)

| Metric                | Value  |
|-----------------------|--------|
| AIC                   | 63,653 |
| Explained deviance    | 13.6%  |
| Gini coefficient      | 0.304  |
| Poisson deviance (unit)| 1.050 |

**Key risk factors (exp(β) — multiplicative effect on claim rate):**

| Feature                  | exp(β) | Interpretation                      |
|--------------------------|--------|-------------------------------------|
| `plan_type = Platinum`   | 1.61×  | Platinum holders file 61% more claims |
| `plan_type = Gold`       | 1.38×  | Gold: 38% more                      |
| `chronic_conditions`     | 1.24×  | Chronic condition: 24% more          |
| `smoker`                 | 1.51×  | Smokers file 51% more claims         |
| `region = London`        | 1.50×  | London: 50% above baseline           |
| `region = Wales`         | 0.67×  | Wales: 33% fewer claims              |

### Gamma GLM (Severity)

| Metric                  | Value    |
|-------------------------|----------|
| AIC                     | 234,428  |
| Explained deviance      | 40.4%    |
| Gini coefficient        | 0.343    |
| Dispersion (φ̂)          | 0.500    |
| Gamma deviance (unit)   | 0.523    |

**Key risk factors for claim costs:**

| Feature               | exp(β) | Interpretation                         |
|-----------------------|--------|----------------------------------------|
| `plan_type = Platinum`| 2.21×  | Platinum claims cost 2.2× more         |
| `chronic_conditions`  | 1.79×  | Chronic: 79% higher cost per claim     |
| `smoker`              | 1.75×  | Smokers' claims cost 75% more          |
| `region = London`     | baseline + 35% vs Wales                |

### Pure Premium (Frequency × Severity)

| Metric           | Value  |
|------------------|--------|
| Gini coefficient | 0.600  |
| MAE              | £3,283 |
| RMSE             | £8,270 |

A Gini of **0.60** on the pure premium means the model meaningfully separates high-risk from low-risk policyholders — well above the random baseline of 0.

---

## Project Structure

```
healthcare-claims-glm/
├── .github/workflows/ci.yml     # CI: lint → test → evaluate → Docker
├── configs/config.yaml          # All parameters in one place
├── data/
│   ├── raw/                     # Input CSV (git-ignored)
│   └── processed/               # Parquet train/test splits
├── models/                      # Serialised model artefacts (git-ignored)
├── notebooks/01_eda.py          # Exploratory data analysis
├── reports/figures/             # 10 publication-quality diagnostic plots
├── scripts/
│   ├── generate_data.py         # Synthetic dataset generation
│   ├── train.py                 # Training pipeline
│   ├── train_mlflow.py          # MLflow-instrumented training
│   └── evaluate.py              # Test-set evaluation + figures
├── src/
│   ├── data/
│   │   ├── loader.py            # Load, validate, split
│   │   └── preprocessor.py     # Feature derivation, parquet I/O
│   ├── features/engineer.py    # sklearn ColumnTransformer pipeline
│   ├── models/
│   │   ├── frequency_model.py  # Poisson GLM
│   │   ├── severity_model.py   # Gamma GLM
│   │   └── pure_premium.py     # Composite model
│   ├── evaluation/metrics.py   # Poisson/Gamma deviance, Gini, calibration
│   └── api/
│       ├── main.py             # FastAPI app
│       └── schemas.py          # Pydantic request/response models
├── tests/                       # 77 pytest tests
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_api.py
├── Dockerfile                   # Multi-stage build
├── docker-compose.yml           # API + MLflow services
├── Makefile                     # make install / data / train / test / serve
└── requirements.txt
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/healthcare-claims-glm.git
cd healthcare-claims-glm

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Generate data, train, evaluate

```bash
make data      # generates 50k synthetic policyholders
make train     # fits Poisson + Gamma GLMs, saves to models/
make evaluate  # test-set metrics + 10 diagnostic figures
```

Or run the MLflow-instrumented version:
```bash
python scripts/train_mlflow.py
mlflow ui      # open http://localhost:5000
```

### 3. Run tests

```bash
make test
# Expected: 77 passed
```

### 4. Serve the API

```bash
make serve
# Opens at http://localhost:8000/docs (Swagger UI)
```

Or with Docker:
```bash
make docker-build
docker-compose up -d
```

---

## API Reference

### `POST /predict`

Predict the expected annual claims cost for a single policyholder.

**Request:**
```json
{
  "age": 45,
  "gender": "Female",
  "bmi": 27.5,
  "smoker": false,
  "chronic_conditions": true,
  "num_dependants": 2,
  "region": "London",
  "plan_type": "Gold",
  "years_as_customer": 5,
  "exposure_years": 1.0
}
```

**Response:**
```json
{
  "pred_frequency": 0.6412,
  "pred_severity": 6013.17,
  "pred_pure_premium": 3855.24,
  "risk_band": "HIGH",
  "model_version": "1.0.0"
}
```

### `POST /predict/batch`

Up to 1,000 policies in a single request.

```json
{
  "policies": [
    { ... },
    { ... }
  ]
}
```

### `GET /health`

Liveness/readiness probe for Kubernetes or load balancer.

### `GET /model/info`

Returns AIC, explained deviance, dispersion, feature names.

---

## Why GLMs (not XGBoost)?

| Property            | XGBoost             | Poisson/Gamma GLM              |
|---------------------|---------------------|--------------------------------|
| Interpretability    | Requires SHAP       | Direct: read exp(β) table      |
| Regulatory filing   | Hard to justify     | Coefficient tables accepted    |
| Distributional fit  | Ignores skew/count  | Designed for these families    |
| Extrapolation       | Poor                | Principled (linear predictor)  |
| Training data size  | Prefers millions    | Works with tens of thousands   |
| Actuary acceptance  | Low                 | Standard industry practice     |

In regulated industries (insurance, banking, healthcare), GLMs are not a fallback — they are the **first choice** precisely because the results are defensible to regulators, auditors, and business stakeholders.

---

## Diagnostic Figures

The evaluation pipeline generates 10 figures in `reports/figures/`:

| File                              | Description                                    |
|-----------------------------------|------------------------------------------------|
| `01_claim_distributions.png`      | Zero-inflated Poisson count + Gamma severity   |
| `02_risk_factor_profiles.png`     | Frequency & severity by region, plan, age band |
| `03_binary_risk_factors.png`      | Smoker, chronic, BMI category effects          |
| `04_continuous_risk_factors.png`  | Age and BMI continuous risk gradients          |
| `05_correlation_matrix.png`       | Numeric feature correlations                   |
| `06_pure_premium_segments.png`    | Business KPI by segment                        |
| `07_lorenz_curves.png`            | Lorenz curves with Gini annotations            |
| `08_calibration.png`              | Predicted vs actual by decile                  |
| `09_diagnostics.png`              | Residuals + top risk factor bar chart          |
| `10a_coef_frequency.png`          | Forest plot — Poisson GLM coefficients         |
| `10b_coef_severity.png`           | Forest plot — Gamma GLM coefficients           |

---

## Skills Demonstrated

| Area               | Technologies / Techniques                                      |
|--------------------|---------------------------------------------------------------|
| Statistical modelling | Poisson GLM, Gamma GLM, IRLS, deviance analysis, Gini     |
| Feature engineering | ColumnTransformer, StandardScaler, OHE, interaction terms   |
| ML experiment tracking | MLflow: params, metrics, artifacts                        |
| API development    | FastAPI, Pydantic v2, batch inference, health probes          |
| Software engineering | Clean architecture, pkl serialisation, config-driven pipeline |
| Testing            | pytest, 77 tests, API integration tests, serialisation tests  |
| DevOps             | Docker multi-stage build, docker-compose, GitHub Actions CI   |
| Data visualisation | 10 publication-quality matplotlib/seaborn figures             |

---

## Extending the Project

- **Tweedie GLM**: fit a single compound Poisson-Gamma model instead of the two-part model
- **Negative Binomial**: if overdispersion violates Poisson assumptions
- **GAMs**: add non-linear spline terms for age/BMI using `pygam`
- **Cross-validation**: k-fold deviance for proper model selection
- **SHAP overlay**: compare GLM coefficients with SHAP values from a gradient-boosted reference model
- **Real data**: replace synthetic data with CMS Open Payments or MIMIC-III

---

## Licence

MIT — see `LICENSE`.

---

*Built by Mickey — targeting Data Science / ML roles in the UK and EU.*
