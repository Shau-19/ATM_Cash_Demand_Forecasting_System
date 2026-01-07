# 🏧 ATM Cash Demand Forecasting System  
**End-to-End Time Series Forecasting | ML Ensemble | Business Impact Modeling**

---

## 📌 Project Overview

Accurate cash forecasting is critical for banks to **minimize cash-outs, reduce idle cash, and optimize replenishment costs**.  
This project builds a **production-ready ATM cash demand forecasting system** using:

- Multi-source financial & socioeconomic data  
- Advanced time-series feature engineering  
- Ensemble machine learning models  
- Backtesting, scenario simulation & cost evaluation  
- A deployable **Streamlit dashboard**

> **Outcome:** A scalable forecasting system that predicts district-level ATM withdrawal demand and supports operational decision-making.

---

## 🎯 Key Highlights

- 📊 **118 months of historical data** across **700+ districts**
- 🧠 **Ensemble ML system**: Random Forest + Gradient Boosting + XGBoost  
- ⏱ **Time-aware evaluation** using rolling-year backtesting  
- 📈 **Test RMSE ≈ 0.0096** (normalized scale)  
- 💰 **Cost-based evaluation** for cash replenishment optimization  
- 🌐 **Deployed on Streamlit Cloud**

---

## 🗂️ Data Sources

Data was **curated, cleaned, and engineered** from multiple public & financial sources:

- **RBI** – ATM usage & cash withdrawal statistics  
- **NPCI** – UPI, card & digital transaction volumes  
- **Census & Economic Surveys** – population, literacy, income  
- **Night-Time Lights (NTL)** – proxy for economic activity  
- **Synthetic feature alignment** for temporal consistency  

> Final dataset: **District × Month master table** with 30+ engineered features.

---

## 🧪 Feature Engineering

### 🔹 Core Transformations
- Temporal alignment (district-month index)
- Outlier treatment & missing value handling
- Scale normalization for stable training

### 🔹 Time-Series Features
- **Lag features:** `lag_1, lag_2, lag_3, lag_6, lag_12`  
- **Rolling stats:** `roll_mean_3, roll_std_3, roll_mean_6, roll_std_6`  
- **Calendar features:** `year, month_num, quarter`

### 🔹 Economic Signals
- Digital payment penetration  
- Cash intensity index  
- Night-time light index  
- ATM density & urbanization ratios  

---

## 🧠 Modeling Approach

### 1️⃣ Baseline  
- Dummy regressor for sanity check

### 2️⃣ Linear Models  
- Linear, Ridge, Lasso, ElasticNet  

### 3️⃣ Tree Models  
- Decision Tree  
- **Random Forest**  
- **Gradient Boosting**

### 4️⃣ Boosted Trees  
- **XGBoost**

### 5️⃣ Final Ensemble

**Ensemble = 0.4 × RandomForest + 0.3 × GradientBoosting + 0.3 × XGBoost**

---

## 📏 Evaluation Strategy

### 🔹 Time-Aware Splitting
- **Train:** ≤ 2021  
- **Validation:** 2022–2023  
- **Test:** ≥ 2024  

### 🔹 Final Performance

| Model | MAE (Test) | RMSE (Test) |
|------|------------|-------------|
| Baseline | 0.0109 | 0.0163 |
| Random Forest | 0.0088 | 0.0133 |
| Gradient Boosting | 0.0089 | 0.0129 |
| XGBoost | 0.0087 | 0.0127 |
| **Ensemble** | **0.0060** | **0.0096** |

---

## 🔁 Backtesting

- Rolling-year evaluation  
- District-wise stability checks  
- Error distribution monitoring  
- Identified **top-performing districts** for dashboard defaults  

---

## 🎭 Scenario Simulation

Stress-tested forecasts under real-world conditions:

- 🎉 **Festival Surge** → +15% cash demand  
- 📱 **Digital Push** → −10% ATM withdrawals  
- 🏦 **Policy Shift** → change in cash intensity  

---

## 💰 Cost-Based Evaluation

Converted predictions into **business KPIs**:

- Cash replenishment frequency  
- Over-stock vs under-stock penalty  
- Idle cash cost  
- Emergency refill risk  

> Shows how ML directly enables **operational savings**.

---

## 🌐 Streamlit Dashboard

### Features
- District selection (best-performing / all)  
- 24-month forecast visualization  
- Ensemble predictions vs actuals  
- Real-time MAE / RMSE metrics  
- Feature-order locking for reliability
- Dashboard Link:  https://shau-19-atm-cash-demand-forecasting-sys-appforecast-demo-rzo1ah.streamlit.app/  

---

## 🗂️ Project Structure
ATM_Cash_Demand_Forecasting_System/
│
├── app/
│   └── forecast_demo.py
│
├── data/
│   ├── master_district_month_clean.csv
│   ├── district_month_MODEL_READY_LAGS.csv
│   └── model_features.txt
│
├── models/
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── xgboost.pkl
│
├── notebooks/
│   ├── data_pipeline.ipynb
│   ├── modeling.ipynb
│   ├── backtesting.ipynb
│   ├── scenario_simulation.ipynb
│   └── cost_evaluation.ipynb
│
├── requirements.txt
└── README.md


---

## ⚙️ Installation & Run

```bash
git clone https://github.com/Shau-19/ATM_Cash_Demand_Forecasting_System.git
cd ATM_Cash_Demand_Forecasting_System

pip install -r requirements.txt
streamlit run app/forecast_demo.py




