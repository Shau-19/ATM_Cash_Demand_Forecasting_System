# 🏧 ATM Cash Demand Forecasting System

**Hybrid Time Series Forecasting | ML Ensemble + Prophet | 746 Districts | ~1% MAPE**

---

## 📌 Project Overview

Accurate cash forecasting is critical for banks to **minimize cash-outs, reduce idle cash, and optimize replenishment costs**.

This project builds a **production-ready ATM cash demand forecasting system** using:

- 118 months of historical district-level data across **746 Indian districts**
- **Hybrid forecasting engine**: Gradient Boosting + XGBoost ensemble blended with Facebook Prophet
- Time-safe training with a strict global date cutoff split
- Pre-trained per-district Prophet models with economic regressors
- A fully interactive **Streamlit dashboard** with future forecasting, confidence intervals, momentum analysis, and trend decomposition

> **Outcome:** Sub-1% MAPE on the hybrid engine for district-level ATM withdrawal demand.

---

## 🎯 Key Highlights

| Metric | Value |
|---|---|
| Districts covered | 746 |
| Historical data | 118 months (2015–2025) |
| ML Model MAPE | ~1.29% |
| Prophet MAPE | ~1.41% |
| **Hybrid MAPE** | **~1.00%** |
| Hybrid MAE | 0.0041 |
| Hybrid RMSE | 0.0055 |

---

## 🗂️ Data Sources

Data was curated, cleaned, and engineered from multiple public and financial sources:

- **RBI** — ATM usage & cash withdrawal statistics
- **NPCI** — UPI, card & digital transaction volumes
- **Census & Economic Surveys** — population, literacy, income
- **Night-Time Lights (NTL)** — proxy for economic activity

> Final dataset: **District × Month master table** with 30+ engineered features.

---

## 🧠 Modeling Approach

### ML Ensemble (GB + XGB)

Two gradient-boosted models trained on the full feature set, averaged with equal weight:

```
ML Prediction = 0.5 × GradientBoosting + 0.5 × XGBoost
```

**Full feature set (27 features):** lag features (lag_1/2/3/6/12), rolling_mean_3, digital payment signals, UPI volume, cash intensity index, ATM density, population, economic indicators, calendar features.

**Strict feature set (23 features):** removes static leakage columns (district_share, num_atms, population_total_k, rolling_mean_3) for a cleaner out-of-sample evaluation.

### Prophet (per-district)

Pre-trained Facebook Prophet model for each of the 746 districts with:

- Multiplicative yearly seasonality
- Economic regressors: `upi_txn_volume_mn`, `digital_share_txn`, `cash_intensity_index`
- `changepoint_prior_scale=0.1` for flexibility on volatile monthly series

### Hybrid Engine

Volatility-adaptive blending:

```
if recent_volatility > 0.1:
    Hybrid = 0.7 × ML + 0.3 × Prophet   # ML-dominant for volatile districts
else:
    Hybrid = 0.5 × ML + 0.5 × Prophet   # Equal blend for stable districts
```

---

## 🔁 Feature Engineering

### Time-Series Features
- **Lag features:** `lag_1, lag_2, lag_3, lag_6, lag_12`
- **Rolling stats:** `rolling_mean_3`
- **Calendar:** `year, month_num`

### Economic Signals
- Digital payment penetration (`digital_share_txn`)
- Cash intensity index
- Night-time light index (`ntl_index_mean`, `ntl_index_latest`)
- ATM density & urbanization ratios (`metro_atm_share`, `urban_atm_share`, `rural_atm_share`)
- UPI, POS, wallet transaction volumes

---

## ⏱ Time-Safe Training Split

```python
cutoff = df["date"].max() - pd.DateOffset(months=12)

train_df = df[df["date"] <= cutoff]   # ~2015–Oct 2024
test_df  = df[df["date"] >  cutoff]   # Nov 2024–Oct 2025
```

Both ML and Prophet models are trained on the same cutoff — no data leakage.

---

## 🌐 Streamlit Dashboard

### Features

- **3 forecast engines:** ML Model, Prophet, Hybrid
- **2 view modes:** Real Scale (mn) and Normalized (0–1)
- **Future forecasting:** 1–12 months ahead with 85% confidence intervals
- **Momentum chart:** Month-over-Month % change (green/red bar chart, always on raw scale)
- **Trend decomposition:** Long-term trend + yearly seasonality from Prophet
- **4 metrics:** MAE, RMSE, MAPE (always real scale), Latest Prediction with delta
- **746 districts** selectable from sidebar
- **Full + Strict ML modes** for comparing leakage-free vs full feature performance

### Run Locally

```bash
git clone https://github.com/Shau-19/ATM_Cash_Demand_Forecasting_System.git
cd ATM_Cash_Demand_Forecasting_System

pip install -r requirements.txt

# Train ML models
python notebooks/train.py

# Train Prophet models (per district, ~746 models)
python notebooks/train_prophet.py

# Launch dashboard
streamlit run app/forecast_demo.py
```

---

## 🗂️ Project Structure

```
ATM_Cash_Demand_Forecasting_System/
│
├── app/
│   └── forecast_demo.py          # Streamlit dashboard
│
├── data/
│   └── final_forecast_data.csv   # District × Month master table
│
├── models/
│   ├── gb_full.pkl               # Gradient Boosting (full features)
│   ├── xgb_full.pkl              # XGBoost (full features)
│   ├── gb_strict.pkl             # Gradient Boosting (strict features)
│   ├── xgb_strict.pkl            # XGBoost (strict features)
│   ├── features_full.json        # Feature list for full model
│   ├── features_strict.json      # Feature list for strict model
│   ├── prophet_meta.json         # Per-district Prophet metadata
│   └── prophet/
│       └── {district}.pkl        # Pre-trained Prophet model per district
│
├── notebooks/
│   ├── train.py                  # ML model training script
│   └── train_prophet.py          # Prophet per-district training script
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Requirements

```
streamlit
pandas
numpy
scikit-learn
xgboost
prophet
joblib
plotly
```

---

## 📊 Dashboard Screenshots

**Hybrid Engine — Real Scale**
Tight predicted vs actual fit with future forecast and 85% CI band.

**Momentum Chart**
Month-over-Month % change always computed on raw withdrawal values.

**Trend Decomposition**
Prophet long-term trend + yearly seasonality clipped to actual data range (2016–2025).

---

## 🔑 Key Design Decisions

| Decision | Reason |
|---|---|
| Global date cutoff split | Prevents district-level data leakage from `groupby().tail()` |
| Pre-trained Prophet loaded in dashboard | Eliminates per-request refitting (major latency + accuracy improvement) |
| Momentum on raw values | Normalized pct_change near zero produces garbage large values |
| Volatility-adaptive hybrid weights | High-volatility districts benefit from ML dominance |
| Strict feature set | Removes static district identity proxies for honest generalization |

---

## 👤 Author

**Shaurya** — BTech 3rd Year, AI/ML Engineer  
[GitHub](https://github.com/Shau-19)
