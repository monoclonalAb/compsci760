"""
lgb_next_location_haversine_per_iter.py

LightGBM baseline for predicting next GPS coordinate (latitude, longitude)
with per-iteration Haversine RMSE (meters) evaluation & plotting.
"""

import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import joblib
from typing import Tuple
import matplotlib.pyplot as plt

# --------- CONFIG ----------
DATA_PATH = "./data/processed_bird_migration.xlsx"   # change if needed
OUT_DIR = "./baseline/lgb_results"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 0
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
    q995 = np.quantile(step_dist_km, 0.995)
    mask = step_dist_km <= q995
    removed = (~mask).sum()
    if removed > 0:
        print(f"Removing {removed} extreme steps > 99.5 percentile (~{q995:.3f} km)")
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
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        extra_feats.append("timestamp")

    def build_features(ddf: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feats = []
        feats.append(ddf[lat_col].astype(float).values.reshape(-1, 1))
        feats.append(ddf[lon_col].astype(float).values.reshape(-1, 1))
        if "route_progress" in extra_feats:
            feats.append(ddf["route_progress"].astype(float).values.reshape(-1,1))
        if "cumulative_distance" in extra_feats:
            feats.append(ddf["cumulative_distance"].astype(float).values.reshape(-1,1))
        if "timestamp" in extra_feats:
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

    # 4) Train two LightGBM regressors (lat and lon) with early stopping and record evals
    models = {}
    evals_results = {}
    for dim_idx, dim_name in enumerate(["next_latitude", "next_longitude"]):
        print(f"\nTraining LightGBM for target: {dim_name} (index {dim_idx})")
        train_set = lgb.Dataset(X_train, label=Y_train[:, dim_idx])
        val_set = lgb.Dataset(X_val, label=Y_val[:, dim_idx], reference=train_set)
        test_set = lgb.Dataset(X_test, label=Y_test[:, dim_idx], reference=train_set)

        params = LGB_PARAMS.copy()

        # prepare an empty dict to be filled by record_evaluation callback
        evals_result = {}

        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result),
        ]

        gbm = lgb.train(params,
                        train_set,
                        num_boost_round=N_ESTIMATORS,
                        valid_sets=[train_set, val_set, test_set],
                        valid_names=['train', 'val', 'test'],
                        callbacks=callbacks)

        models[dim_name] = gbm
        evals_results[dim_name] = evals_result

        model_path = os.path.join(OUT_DIR, f"lgb_{dim_name}.txt")
        gbm.save_model(model_path)
        print("Saved model to:", model_path)

    # 5) Compute per-iteration haversine RMSE (meters) for val & test
    # Determine number of iterations available for each model
    def safe_num_iters(gbm):
        # prefer best_iteration if early stopping happened, else num_trees
        if getattr(gbm, "best_iteration", None):
            return int(gbm.best_iteration)
        try:
            return int(gbm.num_trees())
        except Exception:
            return int(N_ESTIMATORS)

    n_iter_lat = safe_num_iters(models["next_latitude"])
    n_iter_lon = safe_num_iters(models["next_longitude"])
    n_common = min(n_iter_lat, n_iter_lon)
    if n_common <= 0:
        print("No iterations found for both models; skipping per-iteration haversine computation.")
    else:
        print(f"Computing haversine RMSE for iterations 1..{n_common} (this will predict on val & test each iteration)...")
        val_rmse_per_iter = []
        test_rmse_per_iter = []

        # loop iterations; predict with num_iteration=i for both models
        for i in range(1, n_common + 1):
            # predict val
            pred_lat_val = models["next_latitude"].predict(X_val, num_iteration=i)
            pred_lon_val = models["next_longitude"].predict(X_val, num_iteration=i)
            d_val = haversine_meters(Y_val[:,0], Y_val[:,1], pred_lat_val, pred_lon_val)
            rmse_val = float(np.sqrt(np.mean(d_val**2)))
            val_rmse_per_iter.append(rmse_val)

            # predict test
            pred_lat_test = models["next_latitude"].predict(X_test, num_iteration=i)
            pred_lon_test = models["next_longitude"].predict(X_test, num_iteration=i)
            d_test = haversine_meters(Y_test[:,0], Y_test[:,1], pred_lat_test, pred_lon_test)
            rmse_test = float(np.sqrt(np.mean(d_test**2)))
            test_rmse_per_iter.append(rmse_test)

            if (i % 50 == 0) or (i == 1) or (i == n_common):
                print(f" iter {i:04d}: val_haversine_rmse = {rmse_val:.2f} m, test_haversine_rmse = {rmse_test:.2f} m")

        val_rmse_per_iter = np.array(val_rmse_per_iter, dtype=float)
        test_rmse_per_iter = np.array(test_rmse_per_iter, dtype=float)

        # 6) Plot haversine RMSE per iteration
        plt.figure(figsize=(10,5))
        iters = np.arange(1, n_common + 1)
        plt.plot(iters, val_rmse_per_iter, label='Val Haversine RMSE (m)', marker='o')
        plt.plot(iters, test_rmse_per_iter, label='Test Haversine RMSE (m)', marker='x')
        plt.xlabel('Boosting iteration')
        plt.ylabel('Haversine RMSE (meters)')
        plt.title('Haversine RMSE per boosting iteration')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_plot = os.path.join(OUT_DIR, "haversine_rmse_per_iter.png")
        plt.savefig(out_plot, dpi=150)
        plt.close()
        print("Saved haversine RMSE plot to:", out_plot)

        # Save CSV
        df_iters = pd.DataFrame({
            "iter": iters,
            "val_haversine_rmse_m": val_rmse_per_iter,
            "test_haversine_rmse_m": test_rmse_per_iter
        })
        csv_out = os.path.join(OUT_DIR, "haversine_rmse_per_iter.csv")
        df_iters.to_csv(csv_out, index=False)
        print("Saved per-iteration haversine RMSE CSV to:", csv_out)

    # 7) Final predictions on test set using best_iteration for each model and final haversine metrics
    pred_lat = models["next_latitude"].predict(X_test, num_iteration=models["next_latitude"].best_iteration if getattr(models["next_latitude"], "best_iteration", None) else None)
    pred_lon = models["next_longitude"].predict(X_test, num_iteration=models["next_longitude"].best_iteration if getattr(models["next_longitude"], "best_iteration", None) else None)

    metrics = regression_metrics_haversine(Y_test[:,0], Y_test[:,1], pred_lat, pred_lon)
    print("\nFinal Test Haversine metrics (meters) using best_iteration:")
    for k,v in metrics.items():
        print(f"  {k}: {v:.3f}")

    # Also print coordinate RMSE (degrees) for interest
    lat_mse = mean_squared_error(Y_test[:,0], pred_lat)
    lon_mse = mean_squared_error(Y_test[:,1], pred_lon)
    combined_mse_final = (lat_mse + lon_mse) / 2.0
    print(f"\nFinal coordinate-space RMSE (deg): lat {math.sqrt(lat_mse):.6f}, lon {math.sqrt(lon_mse):.6f}")
    print(f"Final combined RMSE (deg): {math.sqrt(combined_mse_final):.6f}")

    # Save models (joblib wrapper)
    joblib.dump(models, os.path.join(OUT_DIR, "lgb_models.pkl"))
    print("Saved LightGBM models (joblib).")

    # Save test-level predictions CSV
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
