"""
lgb_next_location.py

LightGBM baseline for predicting next GPS coordinate (latitude, longitude)
from current position + simple features.

Requirements:
  pip install pandas numpy scikit-learn lightgbm joblib
"""

import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
from typing import Tuple

# --------- CONFIG ----------
DATA_PATH = "./data/processed_bird_migration.xlsx"   # change if needed
OUT_DIR = "./lgb_results"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# LightGBM params (base)
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "seed": RANDOM_SEED,
    "verbose": -1,
}

N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 50

# --------- HELPERS ----------
def haversine_meters(lat1, lon1, lat2, lon2):
    """Vectorized haversine in meters."""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def grouped_route_split(df: pd.DataFrame, route_col: str,
                        train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, test_frac=TEST_FRAC,
                        seed=RANDOM_SEED):
    route_ids = df[route_col].unique().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(route_ids)
    n = len(route_ids)
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    train_ids = route_ids[:n_train]
    val_ids = route_ids[n_train:n_train + n_val]
    test_ids = route_ids[n_train + n_val:]
    train_df = df[df[route_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[route_col].isin(val_ids)].reset_index(drop=True)
    test_df = df[df[route_col].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df, (train_ids, val_ids, test_ids)

def regression_metrics_haversine(lat_true, lon_true, lat_pred, lon_pred) -> dict:
    d = haversine_meters(lat_true, lon_true, lat_pred, lon_pred)
    mse = np.mean(d**2)
    rmse = math.sqrt(mse)
    return {
        "mean_m": float(np.mean(d)),
        "median_m": float(np.median(d)),
        "rmse_m": float(rmse),
        "p75_m": float(np.percentile(d, 75)),
        "p95_m": float(np.percentile(d, 95)),
    }

# --------- MAIN ----------
def main():
    np.random.seed(RANDOM_SEED)

    # 1) Load data
    print("Loading:", DATA_PATH)
    df = pd.read_excel(DATA_PATH)
    print("Rows before dropna:", len(df))

    # Required columns
    lat_col = "GPS_yy"
    lon_col = "GPS_xx"
    next_lat = "next_latitude"
    next_lon = "next_longitude"

    for c in (lat_col, lon_col, next_lat, next_lon):
        if c not in df.columns:
            raise RuntimeError(f"Required column '{c}' not found in the sheet. Columns: {df.columns.tolist()}")

    # drop rows missing essentials
    df = df.dropna(subset=[lat_col, lon_col, next_lat, next_lon]).reset_index(drop=True)
    print("Rows after dropna:", len(df))

    # Optional: remove extreme jumps (outliers) — filter by step distance quantile
    step_dist_km = haversine_meters(df[lat_col].values, df[lon_col].values, df[next_lat].values, df[next_lon].values) / 1000.0
    q95 = np.quantile(step_dist_km, 0.995)    # tune this if needed
    mask = step_dist_km <= q95
    removed = (~mask).sum()
    if removed > 0:
        print(f"Removing {removed} extreme steps > 99.5 percentile (~{q95:.3f} km)")
        df = df[mask].reset_index(drop=True)

    # route column
    if "Migratory route codes" in df.columns:
        route_col = "Migratory route codes"
    elif "ID" in df.columns:
        route_col = "ID"
    else:
        # fallback: create synthetic route grouping by index blocks
        block = 20
        df["_route_tmp"] = (df.index // block).astype(int)
        route_col = "_route_tmp"
        print("No explicit route column found. Created synthetic route column '_route_tmp'.")

    # Optional categorical feature: species (encoded)
    use_species = "Bird species" in df.columns
    if use_species:
        le = LabelEncoder()
        df["__species_le"] = le.fit_transform(df["Bird species"].astype(str).values)
        print("Species found. #classes:", len(le.classes_))
    else:
        print("No species column found; continuing without species feature.")

    # Construct feature matrix X and target Y (rowwise: current -> next)
    extra_feats = []
    if "route_progress" in df.columns:
        extra_feats.append("route_progress")
    if "cumulative_distance" in df.columns:
        extra_feats.append("cumulative_distance")
    # Add time features if present
    if "timestamp" in df.columns:
        # ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        extra_feats.append("timestamp")
    # For simplicity, we'll use: lat, lon, extra numeric features, species if available
    def build_features(ddf: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feats = []
        feats.append(ddf[lat_col].astype(float).values.reshape(-1, 1))
        feats.append(ddf[lon_col].astype(float).values.reshape(-1, 1))
        if "route_progress" in extra_feats:
            feats.append(ddf["route_progress"].astype(float).values.reshape(-1,1))
        if "cumulative_distance" in extra_feats:
            feats.append(ddf["cumulative_distance"].astype(float).values.reshape(-1,1))
        if "timestamp" in extra_feats:
            # encode day-of-year cyclical features
            ts = pd.to_datetime(ddf["timestamp"])
            doy = ts.dt.dayofyear.values
            feats.append(np.sin(2*np.pi*doy/365).reshape(-1,1))
            feats.append(np.cos(2*np.pi*doy/365).reshape(-1,1))
        if use_species:
            feats.append(ddf["__species_le"].values.reshape(-1,1))
        X = np.concatenate(feats, axis=1)
        Y = ddf[[next_lat, next_lon]].astype(float).values
        return X, Y

    # 2) Grouped split by route ids
    train_df, val_df, test_df, (train_ids, val_ids, test_ids) = grouped_route_split(df, route_col, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, RANDOM_SEED)
    print("Split routes -> train/val/test:", len(train_ids), len(val_ids), len(test_ids))
    print("Rows per split:", len(train_df), len(val_df), len(test_df))

    # 3) Build arrays
    X_train, Y_train = build_features(train_df)
    X_val, Y_val = build_features(val_df)
    X_test, Y_test = build_features(test_df)

    print("Feature dim:", X_train.shape[1])
    print("Train samples:", X_train.shape[0], "Val:", X_val.shape[0], "Test:", X_test.shape[0])

    # 4) Train two LightGBM regressors (lat and lon) with early stopping
    models = {}
    for dim_idx, dim_name in enumerate(["next_latitude", "next_longitude"]):
        print(f"\nTraining LightGBM for target: {dim_name} (index {dim_idx})")
        train_set = lgb.Dataset(X_train, label=Y_train[:, dim_idx])
        val_set = lgb.Dataset(X_val, label=Y_val[:, dim_idx], reference=train_set)
        params = LGB_PARAMS.copy()
        # callback
        callbacks = []
        # early stopping via callback
        callbacks.append(lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True))
        # periodic logging (prints evaluation every 50 rounds)
        callbacks.append(lgb.log_evaluation(period=50))
        # fit
        gbm = lgb.train(params,
                        train_set,
                        num_boost_round=N_ESTIMATORS,
                        valid_sets=[train_set, val_set],
                        valid_names=['train','val'],
                        callbacks=callbacks)
        models[dim_name] = gbm
        # save model
        model_path = os.path.join(OUT_DIR, f"lgb_{dim_name}.txt")
        gbm.save_model(model_path)
        print("Saved model to:", model_path)

    # 5) Predict on test set
    print("\nPredicting on test set...")
    pred_lat = models["next_latitude"].predict(X_test, num_iteration=models["next_latitude"].best_iteration)
    pred_lon = models["next_longitude"].predict(X_test, num_iteration=models["next_longitude"].best_iteration)

    # Compute haversine error
    metrics = regression_metrics_haversine(Y_test[:,0], Y_test[:,1], pred_lat, pred_lon)
    print("\nTest Haversine metrics (meters):")
    for k,v in metrics.items():
        print(f"  {k}: {v:.3f}")

    # Also print coordinate RMSE (degrees) for interest
    lat_mse = mean_squared_error(Y_test[:,0], pred_lat)
    lon_mse = mean_squared_error(Y_test[:,1], pred_lon)
    print(f"\nCoordinate-space RMSE (deg): lat {math.sqrt(lat_mse):.6f}, lon {math.sqrt(lon_mse):.6f}")

    # 6) Feature importance (gain) from one model (lat); show top features
    try:
        feat_names = ["lat","lon"]
        if "route_progress" in extra_feats: feat_names.append("route_progress")
        if "cumulative_distance" in extra_feats: feat_names.append("cumulative_distance")
        if "timestamp" in extra_feats:
            feat_names.extend(["doy_sin","doy_cos"])
        if use_species: feat_names.append("__species_le")
        # If dimensions mismatch, trim/pad
        if len(feat_names) != X_train.shape[1]:
            # fallback: generic names
            feat_names = [f"f{i}" for i in range(X_train.shape[1])]
    except Exception:
        feat_names = [f"f{i}" for i in range(X_train.shape[1])]

    print("\nTop feature importances (by gain) for next_latitude:")
    imp = models["next_latitude"].feature_importance(importance_type="gain")
    idx_sorted = np.argsort(-imp)[:20]
    for i in idx_sorted:
        print(f"  {feat_names[i]:20s}  gain={imp[i]:.3f}")

    # Save models (joblib wrapper)
    joblib.dump(models, os.path.join(OUT_DIR, "lgb_models.pkl"))
    print("Saved LightGBM models (joblib).")

    # Save a small results summary CSV
    summary_df = pd.DataFrame({
        "true_lat": Y_test[:,0],
        "true_lon": Y_test[:,1],
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "err_m": haversine_meters(Y_test[:,0], Y_test[:,1], pred_lat, pred_lon)
    })
    summary_df.to_csv(os.path.join(OUT_DIR, "lgb_test_predictions.csv"), index=False)
    print("Saved test predictions and errors to CSV.")

if __name__ == "__main__":
    main()
