"""
hyperparameter_optimization.py

Finds optimal hyperparameters for LightGBM and RNN models using Optuna.
Supports both models with comprehensive search spaces and proper validation.
"""

import os
import numpy as np
import pandas as pd
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
DATA_PATH = "./data/processed_bird_migration.xlsx"
OUTPUT_DIR = "./hyperopt_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_TYPE = "rnn"  # Change to "lightgbm" or "rnn" for optimization
N_TRIALS = 50  # Number of hyperparameter combinations to try
RANDOM_SEED = 0

# Fixed split ratios
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# ==================== HELPER FUNCTIONS ====================
def haversine_meters(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in meters."""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def grouped_route_split(df, route_col, train_frac, val_frac, test_frac, seed):
    """Split data by route IDs to prevent leakage."""
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
    
    return train_df, val_df, test_df

def load_and_prepare_data():
    """Load and preprocess the bird migration data."""
    df = pd.read_excel(DATA_PATH)
    
    # Required columns
    lat_col = "GPS_yy"
    lon_col = "GPS_xx"
    next_lat = "next_latitude"
    next_lon = "next_longitude"
    
    # Drop missing values
    df = df.dropna(subset=[lat_col, lon_col, next_lat, next_lon]).reset_index(drop=True)
    
    # Remove extreme outliers
    step_dist_km = haversine_meters(
        df[lat_col].values, df[lon_col].values,
        df[next_lat].values, df[next_lon].values
    ) / 1000.0
    q995 = np.quantile(step_dist_km, 0.995)
    df = df[step_dist_km <= q995].reset_index(drop=True)
    
    # Determine route column
    if "Migratory route codes" in df.columns:
        route_col = "Migratory route codes"
    elif "ID" in df.columns:
        route_col = "ID"
    else:
        df["_route_tmp"] = (df.index // 20).astype(int)
        route_col = "_route_tmp"
    
    # Identify extra features
    extra_feats = []
    if "route_progress" in df.columns:
        extra_feats.append("route_progress")
    if "cumulative_distance" in df.columns:
        extra_feats.append("cumulative_distance")
    
    return df, lat_col, lon_col, next_lat, next_lon, route_col, extra_feats

# ==================== LIGHTGBM OPTIMIZATION ====================
def objective_lightgbm(trial):
    """Optuna objective function for LightGBM."""
    
    # Load data (cached in practice)
    df, lat_col, lon_col, next_lat, next_lon, route_col, extra_feats = load_and_prepare_data()
    
    # Split data
    train_df, val_df, test_df = grouped_route_split(
        df, route_col, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, RANDOM_SEED
    )
    
    # Build features
    def build_features(ddf):
        feats = [
            ddf[lat_col].astype(float).values.reshape(-1, 1),
            ddf[lon_col].astype(float).values.reshape(-1, 1)
        ]
        for feat in extra_feats:
            feats.append(ddf[feat].astype(float).values.reshape(-1, 1))
        X = np.concatenate(feats, axis=1)
        Y = ddf[[next_lat, next_lon]].astype(float).values
        return X, Y
    
    X_train, Y_train = build_features(train_df)
    X_val, Y_val = build_features(val_df)
    
    # Suggest hyperparameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        "seed": RANDOM_SEED,
        "verbose": -1,
    }
    
    n_estimators = trial.suggest_int("n_estimators", 100, 1500)
    early_stopping = trial.suggest_int("early_stopping_rounds", 30, 100)
    
    # Train models for lat and lon
    haversine_errors = []
    
    for dim_idx in [0, 1]:  # latitude, longitude
        train_set = lgb.Dataset(X_train, label=Y_train[:, dim_idx])
        val_set = lgb.Dataset(X_val, label=Y_val[:, dim_idx], reference=train_set)
        
        gbm = lgb.train(
            params,
            train_set,
            num_boost_round=n_estimators,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        
        if dim_idx == 0:
            pred_lat = pred
        else:
            pred_lon = pred
    
    # Calculate Haversine RMSE on validation set
    distances = haversine_meters(Y_val[:, 0], Y_val[:, 1], pred_lat, pred_lon)
    rmse_meters = float(np.sqrt(np.mean(distances**2)))
    
    return rmse_meters

# ==================== RNN OPTIMIZATION ====================
class TrajStepDataset(Dataset):
    def __init__(self, X_seq, y_loc):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y_loc = torch.tensor(y_loc, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_loc[idx]

class BaselineLSTM(nn.Module):
    def __init__(self, feat_dim, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        loc = self.reg_head(h)
        return loc

def objective_rnn(trial):
    """Optuna objective function for RNN/LSTM."""
    
    # Load data
    df, lat_col, lon_col, next_lat, next_lon, route_col, extra_feats = load_and_prepare_data()
    
    # Split data
    train_df, val_df, test_df = grouped_route_split(
        df, route_col, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, RANDOM_SEED
    )
    
    # Suggest hyperparameters
    seq_len = trial.suggest_int("seq_len", 1, 1)  # Keep at 1 for simplicity
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 30, 150)
    
    # Build rowwise features (seq_len=1)
    def build_rowwise(ddf):
        X = ddf[[lat_col, lon_col] + extra_feats].astype(float).values
        Y = ddf[[next_lat, next_lon]].astype(float).values
        return X.reshape(-1, 1, X.shape[1]), Y
    
    X_train, Y_train = build_rowwise(train_df)
    X_val, Y_val = build_rowwise(val_df)
    
    feat_dim = X_train.shape[2]
    
    # Scale features and targets
    feat_scaler = StandardScaler().fit(X_train.reshape(-1, feat_dim))
    targ_scaler = StandardScaler().fit(Y_train)
    
    X_train_s = feat_scaler.transform(X_train.reshape(-1, feat_dim)).reshape(X_train.shape)
    X_val_s = feat_scaler.transform(X_val.reshape(-1, feat_dim)).reshape(X_val.shape)
    
    Y_train_s = targ_scaler.transform(Y_train)
    Y_val_s = targ_scaler.transform(Y_val)
    
    # Create datasets and loaders
    train_ds = TrajStepDataset(X_train_s, Y_train_s)
    val_ds = TrajStepDataset(X_val_s, Y_val_s)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Build model
    model = BaselineLSTM(feat_dim, hidden_size, num_layers, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    # Training loop
    best_val_rmse = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = mse_loss(pred, yb)
            loss.backward()
            opt.step()
        
        # Validate with Haversine
        model.eval()
        preds_coords = []
        trues_coords = []
        
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                pred_s = model(Xb).cpu().numpy()
                true_s = yb.numpy()
                
                pred_unscaled = targ_scaler.inverse_transform(pred_s)
                true_unscaled = targ_scaler.inverse_transform(true_s)
                
                preds_coords.append(pred_unscaled)
                trues_coords.append(true_unscaled)
        
        preds_coords = np.vstack(preds_coords)
        trues_coords = np.vstack(trues_coords)
        
        distances = haversine_meters(
            trues_coords[:, 0], trues_coords[:, 1],
            preds_coords[:, 0], preds_coords[:, 1]
        )
        val_rmse = float(np.sqrt(np.mean(distances**2)))
        
        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        # Report intermediate values for pruning
        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_rmse

# ==================== MAIN OPTIMIZATION ====================
def main():
    print(f"Starting hyperparameter optimization for {MODEL_TYPE.upper()}")
    print(f"Number of trials: {N_TRIALS}")
    print(f"Random seed: {RANDOM_SEED}")
    print("-" * 60)
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",  # Minimize RMSE
        study_name=f"{MODEL_TYPE}_optimization",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Run optimization
    if MODEL_TYPE == "lightgbm":
        objective_func = objective_lightgbm
    elif MODEL_TYPE == "rnn":
        objective_func = objective_rnn
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    study.optimize(objective_func, n_trials=N_TRIALS, show_progress_bar=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\nBest RMSE (Haversine meters): {study.best_value:.2f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_best_params.txt")
    with open(results_path, "w") as f:
        f.write(f"Best RMSE: {study.best_value:.2f} meters\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to: {results_path}")
    
    # Save study dataframe
    df_study = study.trials_dataframe()
    df_study.to_csv(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_trials.csv"), index=False)
    print(f"Trial history saved to: {OUTPUT_DIR}/{MODEL_TYPE}_trials.csv")
    
    # Create visualizations
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_html(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_optimization_history.html"))
        
        fig2 = plot_param_importances(study)
        fig2.write_html(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_param_importances.html"))
        
        print(f"Visualizations saved to: {OUTPUT_DIR}/")
    except Exception as e:
        print(f"Could not create visualizations: {e}")

if __name__ == "__main__":
    main()