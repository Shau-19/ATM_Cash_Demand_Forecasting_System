import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="ATM Cash Demand Forecasting", layout="wide")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PROPHET_DIR = os.path.join(MODEL_DIR, "prophet")
META_PATH  = os.path.join(MODEL_DIR, "prophet_meta.json")

TARGET = "district_atm_withdrawal_volume_mn"

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "final_forecast_data.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource
def load_ml_models():
    return {
        "full": {
            "gb":  joblib.load(os.path.join(MODEL_DIR, "gb_full.pkl")),
            "xgb": joblib.load(os.path.join(MODEL_DIR, "xgb_full.pkl")),
        },
        "strict": {
            "gb":  joblib.load(os.path.join(MODEL_DIR, "gb_strict.pkl")),
            "xgb": joblib.load(os.path.join(MODEL_DIR, "xgb_strict.pkl")),
        },
    }

@st.cache_resource
def load_features():
    with open(os.path.join(MODEL_DIR, "features_full.json"))   as f: full   = json.load(f)
    with open(os.path.join(MODEL_DIR, "features_strict.json")) as f: strict = json.load(f)
    return full, strict

@st.cache_resource
def load_prophet_model(district: str):
    """Load pre-trained Prophet model for a district."""
    path = os.path.join(PROPHET_DIR, f"{district}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_prophet_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}

df             = load_data()
ml_models      = load_ml_models()
features_full, features_strict = load_features()
prophet_meta   = load_prophet_meta()

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("⚙️ Controls")

all_districts = sorted(df["district_name"].unique())
district = st.sidebar.selectbox(
    "District",
    all_districts,
    help=f"{len(all_districts)} districts available"
)

engine = st.sidebar.radio(
    "Forecast Engine",
    ["ML Model", "Prophet", "Hybrid"]
)

view_mode = st.sidebar.radio(
    "View Mode",
    ["Real Scale", "Normalized"]
)

future_months = st.sidebar.slider("Future Months", 1, 12, 6)

ml_type   = st.sidebar.radio("ML Type", ["Full", "Strict"])
model_key = "full" if ml_type == "Full" else "strict"
features  = features_full if model_key == "full" else features_strict

# Prophet availability check
prophet_available = os.path.exists(os.path.join(PROPHET_DIR, f"{district}.pkl"))
if engine in ["Prophet", "Hybrid"] and not prophet_available:
    st.sidebar.warning(f"⚠️ No pre-trained Prophet model for {district}. Run train_prophet.py first.")

# =====================================
# FILTER DISTRICT  — use last 24 months for display
# =====================================
df_d    = df[df["district_name"] == district].sort_values("date").reset_index(drop=True)
df_plot = df_d.tail(24).copy()

dates    = df_plot["date"]
y_actual_raw = df_plot[TARGET].values   # always keep raw for momentum

# =====================================
# ML PREDICTION
# =====================================
gb  = ml_models[model_key]["gb"]
xgb = ml_models[model_key]["xgb"]

X        = df_plot[features]
ml_pred  = 0.5 * gb.predict(X) + 0.5 * xgb.predict(X)

# =====================================
# PROPHET PREDICTION  — load pre-trained, no refitting
# =====================================
prophet_pred  = np.full(len(df_plot), np.nan)
prophet_upper = np.full(len(df_plot), np.nan)
prophet_lower = np.full(len(df_plot), np.nan)
prophet_future_dates  = pd.DatetimeIndex([])
prophet_future_pred   = np.array([])
prophet_future_upper  = np.array([])
prophet_future_lower  = np.array([])

if prophet_available:
    p_model = load_prophet_model(district)
    meta    = prophet_meta.get(district, {})
    regressors = meta.get("regressors", [])

    # Build future dataframe including history + future months
    last_date  = df_d["date"].max()
    future_df  = p_model.make_future_dataframe(
        periods=future_months + len(df_plot),  # enough to cover display window + future
        freq="MS"
    )

    # Add regressors if model was trained with them
    for col in regressors:
        if col in df_d.columns:
            # Map known dates; fill future with last known value
            reg_series = df_d.set_index("date")[col]
            future_df[col] = future_df["ds"].map(reg_series).fillna(reg_series.iloc[-1])

    forecast = p_model.predict(future_df)
    forecast_indexed = forecast.set_index("ds")

    # Extract values aligned to display window
    for i, d in enumerate(dates):
        if d in forecast_indexed.index:
            prophet_pred[i]  = forecast_indexed.loc[d, "yhat"]
            prophet_upper[i] = forecast_indexed.loc[d, "yhat_upper"]
            prophet_lower[i] = forecast_indexed.loc[d, "yhat_lower"]

    # Future forecast (beyond last known date)
    future_mask = forecast["ds"] > last_date
    prophet_future_dates = pd.to_datetime(forecast.loc[future_mask, "ds"].values[:future_months])
    prophet_future_pred  = forecast.loc[future_mask, "yhat"].values[:future_months]
    prophet_future_upper = forecast.loc[future_mask, "yhat_upper"].values[:future_months]
    prophet_future_lower = forecast.loc[future_mask, "yhat_lower"].values[:future_months]

    # Clip trend decomp to actual data range (always use district data start)
    data_start = df_d["date"].min()
    forecast_clipped = forecast[forecast["ds"] >= data_start].copy()
else:
    forecast_clipped = pd.DataFrame()

# =====================================
# HYBRID PREDICTION
# =====================================
if prophet_available and not np.all(np.isnan(prophet_pred)):
    recent_vol = pd.Series(y_actual_raw).pct_change().std()
    # More weight to ML when volatility is high (ML adapts better)
    ml_weight  = 0.7 if recent_vol > 0.1 else 0.5
    hybrid_pred = ml_weight * ml_pred + (1 - ml_weight) * prophet_pred
    hybrid_upper = ml_weight * ml_pred + (1 - ml_weight) * prophet_upper
    hybrid_lower = ml_weight * ml_pred + (1 - ml_weight) * prophet_lower
else:
    hybrid_pred  = ml_pred.copy()
    hybrid_upper = np.full(len(df_plot), np.nan)
    hybrid_lower = np.full(len(df_plot), np.nan)

# =====================================
# SELECT OUTPUT
# =====================================
if engine == "ML Model":
    pred  = ml_pred.copy()
    upper = np.full(len(df_plot), np.nan)
    lower = np.full(len(df_plot), np.nan)
elif engine == "Prophet":
    pred  = prophet_pred.copy()
    upper = prophet_upper.copy()
    lower = prophet_lower.copy()
else:  # Hybrid
    pred  = hybrid_pred.copy()
    upper = hybrid_upper.copy()
    lower = hybrid_lower.copy()

y_display   = y_actual_raw.copy()
pred_display = pred.copy()
upper_display = upper.copy()
lower_display = lower.copy()
future_pred_display  = prophet_future_pred.copy()
future_upper_display = prophet_future_upper.copy()
future_lower_display = prophet_future_lower.copy()

# =====================================
# NORMALIZATION  — applied AFTER momentum
# =====================================
unit_label = "mn"

if view_mode == "Normalized":
    min_val = y_actual_raw.min()
    max_val = y_actual_raw.max()
    rng     = max_val - min_val + 1e-8

    y_display    = (y_display    - min_val) / rng
    pred_display = (pred_display - min_val) / rng

    valid_ci = ~np.isnan(upper_display)
    upper_display[valid_ci] = (upper_display[valid_ci] - min_val) / rng
    lower_display[valid_ci] = (lower_display[valid_ci] - min_val) / rng

    if len(future_pred_display):
        future_pred_display  = (future_pred_display  - min_val) / rng
        future_upper_display = (future_upper_display - min_val) / rng
        future_lower_display = (future_lower_display - min_val) / rng

    unit_label = "normalized"

# =====================================
# TITLE
# =====================================
st.title("🏧 ATM Cash Demand Forecasting Dashboard")
st.caption(f"District: **{district}** | Engine: **{engine}** | ML Type: **{ml_type}** | Scale: **{view_mode}**")

# =====================================
# MAIN CHART
# =====================================
fig = go.Figure()

# Convert all dates to string for consistent axis rendering
dates_str = [d.strftime("%Y-%m-%d") for d in dates]

# Actual
fig.add_trace(go.Scatter(
    x=dates_str, y=y_display,
    name="Actual",
    mode="lines+markers",
    line=dict(color="#4C9BE8", width=2),
    marker=dict(size=5)
))

# Predicted (historical window)
fig.add_trace(go.Scatter(
    x=dates_str, y=pred_display,
    name="Predicted",
    mode="lines+markers",
    line=dict(color="#FF6B6B", width=2),
    marker=dict(size=5)
))

# Confidence interval (historical)
ci_valid = ~np.isnan(upper_display)
if ci_valid.any():
    ci_dates = dates[ci_valid]
    fig.add_trace(go.Scatter(
        x=[d.strftime("%Y-%m-%d") for d in ci_dates], y=upper_display[ci_valid],
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=[d.strftime("%Y-%m-%d") for d in ci_dates], y=lower_display[ci_valid],
        fill="tonexty",
        name="85% CI (historical)",
        line=dict(width=0),
        fillcolor="rgba(128, 0, 200, 0.25)"
    ))

# Future forecast
if len(prophet_future_dates) > 0 and engine != "ML Model" and prophet_available:
    # Convert to string dates — same format as historical traces to force shared axis
    future_x = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in prophet_future_dates]

    fig.add_trace(go.Scatter(
        x=future_x, y=future_pred_display,
        name="Future Forecast",
        mode="lines+markers",
        line=dict(color="#FFD700", width=2, dash="dot"),
        marker=dict(size=5, symbol="diamond")
    ))

    fig.add_trace(go.Scatter(
        x=future_x, y=future_upper_display,
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=future_x, y=future_lower_display,
        fill="tonexty",
        name="85% CI (future)",
        line=dict(width=0),
        fillcolor="rgba(255, 215, 0, 0.15)"
    ))

    # Add vertical line using shapes+annotation (avoids plotly add_vline bug)
    boundary = df_d["date"].max().strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        x0=boundary, x1=boundary,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="gray", dash="dash", width=1.5)
    )
    fig.add_annotation(
        x=boundary, y=1,
        xref="x", yref="paper",
        text="Forecast →",
        showarrow=False,
        xanchor="left",
        font=dict(color="gray", size=11)
    )

# Force x-axis range to include both historical and future dates on same axis
all_x_dates = list(dates)
if len(prophet_future_dates) > 0 and engine != "ML Model" and prophet_available:
    all_x_dates = list(dates) + list(prophet_future_dates)

x_min = (pd.Timestamp(min(all_x_dates)) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
x_max = (pd.Timestamp(max(all_x_dates)) + pd.DateOffset(months=1)).strftime("%Y-%m-%d")

fig.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_title="Date",
    yaxis_title=f"ATM Withdrawal Volume ({unit_label})",
    xaxis=dict(range=[x_min, x_max], type="date"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================
# METRICS
# =====================================
valid_mask = ~np.isnan(pred_display)
mae  = np.mean(np.abs(y_display[valid_mask] - pred_display[valid_mask]))
rmse = np.sqrt(np.mean((y_display[valid_mask] - pred_display[valid_mask]) ** 2))
mape = np.mean(np.abs((y_actual_raw[valid_mask] - pred[valid_mask]) / (y_actual_raw[valid_mask] + 1e-8))) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE",  f"{mae:.4f}",  help=f"Mean Absolute Error ({unit_label})")
c2.metric("RMSE", f"{rmse:.4f}", help=f"Root Mean Squared Error ({unit_label})")
c3.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute % Error (always on real scale)")
c4.metric(
    "Latest Prediction",
    f"{pred_display[-1]:.4f}",
    delta=f"{pred_display[-1] - pred_display[-2]:.4f}" if len(pred_display) > 1 else None,
    help=f"Most recent prediction ({unit_label})"
)

if view_mode == "Normalized":
    st.caption("⚠️ MAE and RMSE are in normalized scale (0–1). MAPE is always on real scale.")

# =====================================
# MOMENTUM  — ALWAYS computed on raw values  ← FIXED
# =====================================
st.subheader("📈 Momentum (MoM % Change)")

momentum = pd.Series(y_actual_raw).pct_change() * 100   # raw, not normalized

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=dates,
    y=momentum,
    name="MoM %",
    marker_color=np.where(momentum >= 0, "#4CAF50", "#F44336")
))
fig2.add_hline(y=0, line_color="gray", line_dash="dash")
fig2.update_layout(
    template="plotly_dark",
    height=300,
    yaxis_title="MoM Change (%)",
    hovermode="x unified"
)

st.plotly_chart(fig2, use_container_width=True)

# =====================================
# TREND DECOMPOSITION  — clipped to actual data range  ← FIXED
# =====================================
st.subheader("📊 Trend Decomposition (Prophet)")

if prophet_available and not forecast_clipped.empty:
    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=forecast_clipped["ds"],
        y=forecast_clipped["trend"],
        name="Trend",
        line=dict(color="#4C9BE8", width=2)
    ))

    if "yearly" in forecast_clipped.columns:
        fig3.add_trace(go.Scatter(
            x=forecast_clipped["ds"],
            y=forecast_clipped["yearly"],
            name="Yearly Seasonality",
            line=dict(color="#FF6B6B", width=1.5)
        ))

    if "weekly" in forecast_clipped.columns:
        fig3.add_trace(go.Scatter(
            x=forecast_clipped["ds"],
            y=forecast_clipped["weekly"],
            name="Weekly Seasonality",
            line=dict(color="#FFD700", width=1.5)
        ))

    fig3.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Date",
        yaxis_title="Component Value",
        hovermode="x unified"
    )

    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Trend shows long-term growth direction. Seasonality shows repeating patterns within a year.")
else:
    st.info("Trend decomposition requires a pre-trained Prophet model. Run train_prophet.py first.")

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.caption("Hybrid Forecasting System | ML (GB + XGB) + Prophet | Production Ready 🚀")
