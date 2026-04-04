import pandas as pd
import numpy as np
import joblib
import os
import json
from prophet import Prophet

# =====================================
# PATHS
# =====================================
DATA_PATH  = "data/final_forecast_data.csv"
MODEL_DIR  = "models/prophet"
META_PATH  = "models/prophet_meta.json"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["district_name", "date"]).reset_index(drop=True)

TARGET = "district_atm_withdrawal_volume_mn"

EXTRA_REGRESSORS = [
    "upi_txn_volume_mn",
    "digital_share_txn",
    "cash_intensity_index",
]

# =====================================
# TIME-SAFE SPLIT  (same cutoff as ML)
# =====================================
cutoff = df["date"].max() - pd.DateOffset(months=12)

# =====================================
# TRAIN PROPHET PER DISTRICT
# =====================================
districts = df["district_name"].unique()
print(f"Training Prophet for {len(districts)} districts...")
print(f"Train cutoff: {cutoff.date()}\n")

trained  = 0
skipped  = 0
failed   = []
meta     = {}   # store per-district date range for dashboard clipping

for district in districts:
    df_d = df[df["district_name"] == district].copy()

    # Use only training data for fitting
    df_train = df_d[df_d["date"] <= cutoff].copy()

    if len(df_train) < 18:
        skipped += 1
        continue

    # Prophet format
    prophet_df = df_train[["date", TARGET]].rename(columns={
        "date": "ds",
        TARGET: "y"
    })

    # =====================================
    # MODEL — tuned changepoint + seasonality
    # =====================================
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,       # slightly more flexible than default 0.05
        seasonality_prior_scale=5.0,
        interval_width=0.85
    )

    # Add regressors that exist
    available_regressors = []
    for col in EXTRA_REGRESSORS:
        if col in df_train.columns and df_train[col].notna().all():
            prophet_df[col] = df_train[col].values
            model.add_regressor(col)
            available_regressors.append(col)

    try:
        model.fit(prophet_df)

        save_path = os.path.join(MODEL_DIR, f"{district}.pkl")
        joblib.dump(model, save_path)

        meta[district] = {
            "train_start"  : df_train["date"].min().strftime("%Y-%m-%d"),
            "train_end"    : df_train["date"].max().strftime("%Y-%m-%d"),
            "data_start"   : df_d["date"].min().strftime("%Y-%m-%d"),
            "data_end"     : df_d["date"].max().strftime("%Y-%m-%d"),
            "n_train"      : len(df_train),
            "regressors"   : available_regressors,
        }

        trained += 1

        if trained % 50 == 0:
            print(f"  {trained}/{len(districts)} done...")

    except Exception as e:
        print(f"  ❌ {district}: {e}")
        failed.append(district)
        skipped += 1

# Save meta
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print("\n=================================")
print(f"✅ Trained : {trained}")
print(f"⚠️  Skipped : {skipped}")
if failed:
    print(f"❌ Failed  : {failed[:10]}")
print(f"Meta saved: {META_PATH}")
print("=================================")