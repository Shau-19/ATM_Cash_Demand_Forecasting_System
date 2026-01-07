import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
import os, sys 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(
    page_title="ATM Cash Demand Forecasting",
    layout="wide"
)
st.set_option("client.showErrorDetails", False)
warnings.filterwarnings("ignore")

TARGET = "district_atm_withdrawal_volume_mn"

# =====================================
# LOAD FEATURE ORDER (CRITICAL)
# =====================================
@st.cache_resource
def load_feature_order():
    with open(os.path.join(DATA_DIR,"model_features.txt")) as f:
        return [line.strip() for line in f.readlines()]

FEATURES = load_feature_order()

# =====================================
# LOAD METADATA (WITH DISTRICT)
# =====================================
@st.cache_data
def load_metadata():
    df = pd.read_csv(os.path.join(DATA_DIR,"master_district_month_clean.csv"))
    df["date"] = pd.to_datetime(
        df["year"].astype(str)
        + "-"
        + df["month_num"].astype(str).str.zfill(2)
        + "-01"
    )
    return df

# =====================================
# LOAD MODEL FEATURES
# =====================================
@st.cache_data
def load_model_data():
    df = pd.read_csv(os.path.join(DATA_DIR,"district_month_MODEL_READY_LAGS.csv"))

    # Memory optimization
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    return df

# =====================================
# LOAD MODELS
# =====================================
@st.cache_resource
def load_models():
    rf = joblib.load(os.path.join(MODEL_DIR,"random_forest.pkl"))
    gb = joblib.load(os.path.join(MODEL_DIR,"gradient_boosting.pkl"))
    xgb = joblib.load(os.path.join(MODEL_DIR,"xgboost.pkl"))
    return rf, gb, xgb

meta = load_metadata()
model_df = load_model_data()
rf, gb, xgb = load_models()

# =====================================
# COMPUTE DISTRICT-WISE RECENT ERROR
# =====================================
@st.cache_data
def compute_district_errors(
    meta,
    model_df,
    _rf,
    _gb,
    _xgb,
    FEATURES
):
    rows = []

    for district in meta["district_name"].unique():
        meta_slice = (
            meta[meta["district_name"] == district]
            .sort_values("date")
            .tail(24)
        )

        if len(meta_slice) < 12:
            continue

        
        idx = meta_slice.index.intersection(model_df.index)

        if len(idx) < 12:
            continue

        X = model_df.loc[idx, FEATURES]
        y = meta_slice.loc[idx, TARGET].values

        pred = (
            0.4 * _rf.predict(X)
            + 0.3 * _gb.predict(X)
            + 0.3 * _xgb.predict(X)
        )

        mae = np.mean(np.abs(y - pred))
        rows.append((district, mae))

    return pd.DataFrame(rows, columns=["district_name", "mae"])



error_df = compute_district_errors(meta, model_df, rf, gb, xgb, FEATURES)

best_districts = (
    error_df.sort_values("mae")
    .head(30)["district_name"]
    .tolist()
)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("🔍 Controls")

mode = st.sidebar.radio(
    "District Selection Mode",
    ["Best-performing", "All districts"]
)

district_list = (
    best_districts
    if mode == "Best-performing"
    else sorted(meta["district_name"].unique())
)

district = st.sidebar.selectbox(
    "Select District",
    district_list
)

years = (
    meta.loc[meta["district_name"] == district, "year"]
    .sort_values()
    .unique()
)

year = st.sidebar.selectbox(
    "Forecast up to Year",
    years,
    index=len(years) - 1
)

# =====================================
# SAFE TEMPORAL FILTERING
# =====================================
meta_slice = (
    meta[meta["district_name"] == district]
    .sort_values("date")
)

meta_slice = meta_slice[meta_slice["year"] <= year].tail(24)

if meta_slice.empty:
    st.warning("No data available for this selection.")
    st.stop()

time_keys = meta_slice[["year", "month_num"]]

model_slice = (
    model_df.merge(time_keys, on=["year", "month_num"], how="inner")
    .sort_values(["year", "month_num"])
    .tail(len(meta_slice))
)

if model_slice.empty:
    st.warning("Model features missing for selected period.")
    st.stop()

# =====================================
# MODEL INPUT
# =====================================
X_plot = model_slice[FEATURES]
y_actual = meta_slice[TARGET].values
dates = meta_slice["date"]

# =====================================
# PREDICTIONS
# =====================================
rf_pred = rf.predict(X_plot)
gb_pred = gb.predict(X_plot)
xgb_pred = xgb.predict(X_plot)

ensemble_pred = (
    0.4 * rf_pred +
    0.3 * gb_pred +
    0.3 * xgb_pred
)

# =====================================
# UI HEADER
# =====================================
st.title("🏧 ATM Cash Demand Forecasting System")

st.markdown(
    f"""
**District:** {district.title()}  
**Forecast Window:** Last {len(dates)} months  
**Models:** Random Forest + Gradient Boosting + XGBoost
"""
)

# =====================================
# PLOT
# =====================================
fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(dates, y_actual, marker="o", label="Actual (normalized)")
ax.plot(dates, ensemble_pred, marker="x", linestyle="--", label="Predicted")

ax.set_title("ATM Cash Demand Forecast")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized Withdrawal Volume")
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=45)

st.pyplot(fig)

# =====================================
# METRICS
# =====================================
mae = np.mean(np.abs(y_actual - ensemble_pred))
rmse = np.sqrt(np.mean((y_actual - ensemble_pred) ** 2))
relative_error = mae / (np.mean(y_actual) + 1e-6)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE (Normalized)", f"{mae:.4f}")
c2.metric("RMSE (Normalized)", f"{rmse:.4f}")
c3.metric("Latest Prediction", f"{ensemble_pred[-1]:.4f}")
c4.metric("Relative Error (%)", f"{relative_error*100:.1f}%")

st.caption(
    "Forecasts generated using ensemble of Random Forest, Gradient Boosting and XGBoost "
    "with lagged & rolling time-series features. Backtested using rolling-year evaluation."
)
