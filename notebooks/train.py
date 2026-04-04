import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================
# PATHS
# =====================================
DATA_PATH = "data/final_forecast_data.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["district_name", "date"]).reset_index(drop=True)

# =====================================
# CONFIG
# =====================================
TARGET = "district_atm_withdrawal_volume_mn"
TARGET_REAL = "district_atm_withdrawal_value_cr"

DROP_COLS = [
    "district_name",
    "date",
    "state_name",
    "month",
    TARGET,
    TARGET_REAL,
    # drop raw aggregates — these are leakage
    "withdrawal_volume_million",
    "withdrawal_value_crore",
]

FEATURES = [col for col in df.columns if col not in DROP_COLS]

# strict: also drop static/slow-changing cols that could leak district identity
DROP_MORE = [
    "district_share",
    "num_atms",
    "population_total_k",
    "rolling_mean_3",   # computed from target — leakage in strict sense
]
FEATURES_STRICT = [f for f in FEATURES if f not in DROP_MORE]

print(f"FULL features  : {len(FEATURES)}")
print(f"STRICT features: {len(FEATURES_STRICT)}")
print(f"FULL   : {FEATURES}")
print(f"STRICT : {FEATURES_STRICT}")

# =====================================
# TIME-SAFE SPLIT  ← FIXED
# Use a global date cutoff, not head/tail per group
# Last 12 months of data = test
# =====================================
cutoff = df["date"].max() - pd.DateOffset(months=12)

train_df = df[df["date"] <= cutoff].copy()
test_df  = df[df["date"] >  cutoff].copy()

print(f"\nTrain: {train_df['date'].min().date()} → {train_df['date'].max().date()}  ({len(train_df)} rows)")
print(f"Test : {test_df['date'].min().date()}  → {test_df['date'].max().date()}  ({len(test_df)} rows)")
print(f"Test districts: {test_df['district_name'].nunique()}")

# =====================================
# PREPARE SETS
# =====================================
X_train    = train_df[FEATURES]
y_train    = train_df[TARGET]
X_test     = test_df[FEATURES]
y_test     = test_df[TARGET]

X_train_s  = train_df[FEATURES_STRICT]
X_test_s   = test_df[FEATURES_STRICT]

# =====================================
# TRAIN MODELS
# =====================================
print("\nTraining FULL models...")

gb_full = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

xgb_full = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

gb_full.fit(X_train, y_train)
xgb_full.fit(X_train, y_train)

print("Training STRICT models...")

gb_strict = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

xgb_strict = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

gb_strict.fit(X_train_s, y_train)
xgb_strict.fit(X_train_s, y_train)

# =====================================
# EVALUATION
# =====================================
def evaluate(X, y, gb, xgb, label):
    pred = 0.5 * gb.predict(X) + 0.5 * xgb.predict(X)
    mae  = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mape = np.mean(np.abs((y - pred) / (y + 1e-8))) * 100
    print(f"\n{label}")
    print(f"  MAE  : {mae:.6f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAPE : {mape:.2f}%")
    return pred

pred_full   = evaluate(X_test,   y_test, gb_full,   xgb_full,   "FULL MODEL")
pred_strict = evaluate(X_test_s, y_test, gb_strict, xgb_strict, "STRICT MODEL")

# =====================================
# FEATURE IMPORTANCE
# =====================================
importances = pd.Series(
    0.5 * gb_full.feature_importances_ + 0.5 * xgb_full.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print("\nTop 10 features (FULL):")
print(importances.head(10).to_string())

# =====================================
# SAVE
# =====================================
joblib.dump(gb_full,   os.path.join(MODEL_DIR, "gb_full.pkl"))
joblib.dump(xgb_full,  os.path.join(MODEL_DIR, "xgb_full.pkl"))
joblib.dump(gb_strict, os.path.join(MODEL_DIR, "gb_strict.pkl"))
joblib.dump(xgb_strict,os.path.join(MODEL_DIR, "xgb_strict.pkl"))

with open(os.path.join(MODEL_DIR, "features_full.json"), "w") as f:
    json.dump(FEATURES, f, indent=2)

with open(os.path.join(MODEL_DIR, "features_strict.json"), "w") as f:
    json.dump(FEATURES_STRICT, f, indent=2)

print("\n✅ All models saved to", MODEL_DIR)