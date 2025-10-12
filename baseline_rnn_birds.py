# baseline_rnn_birds_with_epoch_test_eval.py
# (based on your original baseline_rnn_birds.py with added per-epoch test evaluation + RMSE plot)
import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # <- added for plotting

# -----------------------
# Config (same as yours)
# -----------------------
DATA_PATH = "./data/processed_bird_migration.xlsx"   # change if needed
MODEL_PATH = "./baseline_results/rnn_results/model/baseline_rnn_birds.pth"
SCALER_PATH = "./baseline_results/rnn_results/model/baseline_scalers.npz"
OUT_DIR = "./baseline_results/rnn_results"

SEQ_LEN = 1
BATCH_SIZE = 128
HIDDEN_SIZE = 32
NUM_LAYERS = 1
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 0

# Fix Python built-in RNG
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Fix NumPy RNG
np.random.seed(RANDOM_SEED)

# Fix PyTorch RNGs (CPU + CUDA)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# Ensure deterministic behavior for cuDNN (used by LSTM)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Enforce deterministic algorithms in PyTorch where possible
torch.use_deterministic_algorithms(True)


# -----------------------
# Helpers (unchanged)
# -----------------------
def haversine_np(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in meters between points in degrees.
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def load_and_check_columns(df):
    cur_lon = "GPS_xx"
    cur_lat = "GPS_yy"
    next_lon = "next_longitude"
    next_lat = "next_latitude"

    for c in (cur_lon, cur_lat, next_lon, next_lat):
        if c not in df.columns:
            raise RuntimeError(f"Required column '{c}' not found in the sheet. Columns: {df.columns.tolist()}")

    extra_feats = []
    if "route_progress" in df.columns:
        extra_feats.append("route_progress")
    if "cumulative_distance" in df.columns:
        extra_feats.append("cumulative_distance")

    if "Migratory route codes" in df.columns:
        route_col = "Migratory route codes"
    elif "ID" in df.columns:
        route_col = "ID"
    else:
        route_col = None

    return cur_lat, cur_lon, next_lat, next_lon, extra_feats, route_col

# Dataset and model classes same as yours
class TrajStepDataset(Dataset):
    def __init__(self, X_seq, y_loc, y_sp=None):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y_loc = torch.tensor(y_loc, dtype=torch.float32)
        self.y_sp = torch.tensor(y_sp, dtype=torch.long) if y_sp is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y_sp is None:
            return self.X[idx], self.y_loc[idx], -1
        return self.X[idx], self.y_loc[idx], self.y_sp[idx]

class BaselineLSTM(nn.Module):
    def __init__(self, feat_dim, hidden_size=64, num_layers=1, num_species=0):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden_size, num_layers, batch_first=True)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 2)
        )
        if num_species > 0:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, num_species)
            )
        else:
            self.cls_head = None

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        loc = self.reg_head(h)
        sp_logits = self.cls_head(h) if self.cls_head is not None else None
        return loc, sp_logits

def create_sliding_windows_per_route(df, route_col, lat_col, lon_col, seq_len=8, stride=1, extras=None, sort_col="route_progress"):
    X_list, Y_list, meta = [], [], []
    for rid, g in df.groupby(route_col):
        if sort_col in g.columns:
            g = g.sort_values(sort_col)
        else:
            g = g.sort_index()
        coords = g[[lat_col, lon_col]].values
        extras_arr = g[extras].values if extras else None
        L = coords.shape[0]
        if L < seq_len + 1:
            continue
        for i in range(0, L - seq_len, stride):
            seq_coords = coords[i:i+seq_len]
            if extras_arr is not None:
                seq_extras = extras_arr[i:i+seq_len]
                seq = np.concatenate([seq_coords, seq_extras], axis=1)
            else:
                seq = seq_coords
            target = coords[i+seq_len]
            if np.isnan(seq).any() or np.isnan(target).any():
                continue
            X_list.append(seq)
            Y_list.append(target)
            meta.append((rid, i))
    if len(X_list) == 0:
        return None, None, None
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y, meta

# -----------------------
# Main training flow with per-epoch test evaluation + plotting
# -----------------------
def main():
    np.random.seed(RANDOM_SEED)
    df = pd.read_excel(DATA_PATH)
    lat_col, lon_col, next_lat, next_lon, extra_feats, route_col = load_and_check_columns(df)

    df = df.dropna(subset=[lat_col, lon_col, next_lat, next_lon]).reset_index(drop=True)

    if route_col is None:
        block_size = max(SEQ_LEN+1, 10)
        df["_route_tmp"] = (df.index // block_size).astype(int)
        route_col = "_route_tmp"

    route_ids = np.array(df[route_col].unique())
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(route_ids)
    n = len(route_ids)
    train_frac, val_frac, test_frac = 0.7, 0.15, 0.15
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    train_ids = route_ids[:n_train]
    val_ids = route_ids[n_train:n_train + n_val]
    test_ids = route_ids[n_train + n_val:]

    train_df = df[df[route_col].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df[route_col].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df[route_col].isin(test_ids)].reset_index(drop=True)

    print(f"Routes total={n}, train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print("Rows per split:", len(train_df), len(val_df), len(test_df))

    X_train, Y_train, meta_train = create_sliding_windows_per_route(train_df, route_col, lat_col, lon_col, seq_len=SEQ_LEN, extras=extra_feats)
    X_val,   Y_val,   meta_val   = create_sliding_windows_per_route(val_df,   route_col, lat_col, lon_col, seq_len=SEQ_LEN, extras=extra_feats)
    X_test,  Y_test,  meta_test  = create_sliding_windows_per_route(test_df,  route_col, lat_col, lon_col, seq_len=SEQ_LEN, extras=extra_feats)

    if X_train is None:
        def rowwise_arrays(ddf):
            Xr = ddf[[lat_col, lon_col] + extra_feats].astype(float).values
            Yr = ddf[[next_lat, next_lon]].astype(float).values
            return Xr, Yr
        X_train, Y_train = rowwise_arrays(train_df)
        X_val,   Y_val   = rowwise_arrays(val_df)
        X_test,  Y_test  = rowwise_arrays(test_df)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val   = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        X_test  = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        meta_train = [(rid, idx) for idx, rid in enumerate(train_df[route_col].values)]
        meta_val = [(rid, idx) for idx, rid in enumerate(val_df[route_col].values)]
        meta_test = [(rid, idx) for idx, rid in enumerate(test_df[route_col].values)]

    print("Sequence shapes (train/val/test):", 
          None if X_train is None else X_train.shape, 
          None if X_val is None else X_val.shape,
          None if X_test is None else X_test.shape)

    feat_dim = X_train.shape[2]
    feat_scaler = StandardScaler().fit(X_train.reshape(-1, feat_dim))
    targ_scaler = StandardScaler().fit(Y_train)

    X_train_s = feat_scaler.transform(X_train.reshape(-1, feat_dim)).reshape(X_train.shape)
    X_val_s   = feat_scaler.transform(X_val.reshape(-1, feat_dim)).reshape(X_val.shape)
    X_test_s  = feat_scaler.transform(X_test.reshape(-1, feat_dim)).reshape(X_test.shape)

    Y_train_s = targ_scaler.transform(Y_train)
    Y_val_s   = targ_scaler.transform(Y_val)
    Y_test_s  = targ_scaler.transform(Y_test)

    species_col = "Bird species" if "Bird species" in df.columns else None
    if species_col:
        le = LabelEncoder()
        le.fit(df[species_col].astype(str).values)
        def build_species_for_meta(meta_list, split_df):
            species_for_seq = []
            route_to_species = split_df.groupby(route_col)[species_col].agg(lambda s: s.mode().iloc[0] if len(s.mode())>0 else s.iloc[0]).to_dict()
            for rid, _ in meta_list:
                sp = route_to_species.get(rid, None)
                if sp is None:
                    species_for_seq.append(0)
                else:
                    species_for_seq.append(int(le.transform([str(sp)])[0]))
            return np.array(species_for_seq, dtype=np.int64)
        sp_train = build_species_for_meta(meta_train, train_df)
        sp_val   = build_species_for_meta(meta_val, val_df)
        sp_test  = build_species_for_meta(meta_test, test_df)
        if (sp_train == -1).any() or (sp_val == -1).any() or (sp_test == -1).any():
            most_common = int(le.transform([df[species_col].mode().iloc[0]])[0])
            sp_train[sp_train == -1] = most_common
            sp_val[sp_val == -1] = most_common
            sp_test[sp_test == -1] = most_common
        num_species = len(le.classes_)
    else:
        le = None
        sp_train = sp_val = sp_test = None
        num_species = 0

    train_ds = TrajStepDataset(X_train_s, Y_train_s, sp_train)
    val_ds = TrajStepDataset(X_val_s, Y_val_s, sp_val)
    test_ds = TrajStepDataset(X_test_s, Y_test_s, sp_test)

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, generator=g)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, generator=g)

    model = BaselineLSTM(feat_dim, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_species=num_species).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    celoss = nn.CrossEntropyLoss() if num_species>0 else None

    def evaluate(loader):
        model.eval()
        total_mse = 0.0; total_ce = 0.0; n = 0
        with torch.no_grad():
            for Xb, ylocb, yspb in loader:
                Xb = Xb.to(DEVICE); ylocb = ylocb.to(DEVICE)
                pred_loc, pred_sp = model(Xb)
                total_mse += mse(pred_loc, ylocb).item() * Xb.size(0)
                if celoss is not None:
                    total_ce += celoss(pred_sp, yspb.to(DEVICE)).item() * Xb.size(0)
                n += Xb.size(0)
        return total_mse / n, (total_ce / n) if celoss is not None else None

    # NEW: evaluate on loader but produce Haversine distances (unscaled coords) and return summary stats
    def evaluate_haversine(loader):
        """
        Runs the model on `loader`, inverse-transforms predicted and true targets,
        computes Haversine distances (meters) and returns:
            (mean_m, median_m, rmse_m, p25, p75, p95, n_samples)
        """
        model.eval()
        preds_coords = []
        trues_coords = []
        n_total = 0
        with torch.no_grad():
            for Xb, ylocb, yspb in loader:
                Xb = Xb.to(DEVICE)
                pred_loc_s, _ = model(Xb)  # scaled predictions
                pred_loc_np = pred_loc_s.cpu().numpy()
                true_loc_np = ylocb.cpu().numpy()
                pred_unscaled = targ_scaler.inverse_transform(pred_loc_np)
                true_unscaled = targ_scaler.inverse_transform(true_loc_np)
                preds_coords.append(pred_unscaled)
                trues_coords.append(true_unscaled)
                n_total += Xb.size(0)
        if n_total == 0:
            return None  # empty set
        preds_coords = np.vstack(preds_coords)
        trues_coords = np.vstack(trues_coords)
        # haversine expects lat1, lon1, lat2, lon2 in degrees
        dists = haversine_np(trues_coords[:,0], trues_coords[:,1], preds_coords[:,0], preds_coords[:,1])
        mean_m = float(np.mean(dists))
        median_m = float(np.median(dists))
        rmse_m = float(np.sqrt(np.mean(dists**2)))
        p25, p75, p95 = [float(np.percentile(dists, q)) for q in (25, 75, 95)]
        return mean_m, median_m, rmse_m, p25, p75, p95, n_total

    # Lists to track RMSE per epoch (for plotting)
    test_rmse_per_epoch = []
    val_rmse_per_epoch = []

    # Prepare CSV logging
    metrics_csv_path = os.path.join(OUT_DIR, "epoch_metrics.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(metrics_csv_path, "w") as f:
        f.write(
            "epoch,train_mse,train_ce,val_mse,val_ce,"
            "val_mean_m,val_median_m,val_rmse_m,val_p25,val_p75,val_p95,"
            "test_mean_m,test_median_m,test_rmse_m,test_p25,test_p75,test_p95\n"
        )

    # training loop (modified to run test evaluation every epoch)
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_mse = 0.0; running_ce = 0.0; tot = 0
        for Xb, ylocb, yspb in train_loader:
            Xb = Xb.to(DEVICE); ylocb = ylocb.to(DEVICE)
            opt.zero_grad()
            pred_loc, pred_sp = model(Xb)
            loss = mse(pred_loc, ylocb)
            if celoss is not None:
                loss = loss + celoss(pred_sp, yspb.to(DEVICE))
            loss.backward()
            opt.step()
            bs = Xb.size(0)
            running_mse += mse(pred_loc, ylocb).item() * bs
            if celoss is not None:
                running_ce += celoss(pred_sp, yspb.to(DEVICE)).item() * bs
            tot += bs
        train_mse = running_mse / tot
        train_ce = (running_ce / tot) if celoss is not None else np.nan

        # validation (scaled MSE for monitoring)
        val_mse, val_ce = evaluate(val_loader)
        if val_ce is None:
            val_ce = np.nan

        # Haversine evaluation on val/test
        val_hav = evaluate_haversine(val_loader)
        test_hav = evaluate_haversine(test_loader)

        # Extract haversine metrics or NaNs if not available
        def unpack_hav(hav):
            if hav is None:
                return (np.nan,)*7
            return hav

        val_mean_m, val_median_m, val_rmse_m, val_p25, val_p75, val_p95, _ = unpack_hav(val_hav)
        test_mean_m, test_median_m, test_rmse_m, test_p25, test_p75, test_p95, _ = unpack_hav(test_hav)

        # record RMSE for plotting
        test_rmse_per_epoch.append(test_rmse_m)
        val_rmse_per_epoch.append(val_rmse_m)

        # print nicely
        msg = (f"Epoch {epoch:02d} | Train MSE: {train_mse:.6f}" +
               (f", Train CE: {train_ce:.4f}" if not np.isnan(train_ce) else "") +
               f" | Val MSE: {val_mse:.6f}" +
               (f", Val CE: {val_ce:.4f}" if not np.isnan(val_ce) else ""))
        if not np.isnan(val_rmse_m):
            msg += f" | Val RMSE: {val_rmse_m:.2f} m"
        if not np.isnan(test_rmse_m):
            msg += f" | Test RMSE: {test_rmse_m:.2f} m"
        print(msg)

        # 🔸 Write metrics for this epoch to CSV
        with open(metrics_csv_path, "a") as f:
            f.write(
                f"{epoch},{train_mse},{train_ce},{val_mse},{val_ce},"
                f"{val_mean_m},{val_median_m},{val_rmse_m},{val_p25},{val_p75},{val_p95},"
                f"{test_mean_m},{test_median_m},{test_rmse_m},{test_p25},{test_p75},{test_p95}\n"
            )

    # save model + scalers + classes (same as yours)
    save_dict = {
        "model_state_dict": model.state_dict(),
        "feat_scaler_mean": feat_scaler.mean_,
        "feat_scaler_var": feat_scaler.var_,
        "target_scaler_mean": targ_scaler.mean_,
        "target_scaler_var": targ_scaler.var_
    }
    if le is not None:
        save_dict["label_classes"] = le.classes_.tolist()

    torch.save(save_dict, MODEL_PATH)
    np.savez(SCALER_PATH, feat_mean=feat_scaler.mean_, feat_var=feat_scaler.var_,
             targ_mean=targ_scaler.mean_, targ_var=targ_scaler.var_)
    print("Saved model:", MODEL_PATH)
    print("Saved scalers:", SCALER_PATH)

    # Plot RMSE per epoch (test)
    epochs = np.arange(1, EPOCHS+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, test_rmse_per_epoch, marker='o', label="Test RMSE (m)")
    # optional: also plot val RMSE
    if any(~np.isnan(val_rmse_per_epoch)):
        plt.plot(epochs, val_rmse_per_epoch, marker='x', label="Val RMSE (m)")
    plt.xlabel("Epoch")
    plt.ylabel("Haversine RMSE (meters)")
    plt.title("RMSE per epoch (Haversine meters)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rmse_per_epoch.png"), dpi=150)
    print("Saved RMSE plot: rmse_per_epoch.png")
    plt.show()

    # (the rest of your final test-evaluation / preview can remain; omitted for brevity)
    # You can still keep the final test summary as in your original script if desired.

    # -----------------------
    # Final evaluation on TEST set (unseen trajectories)
    # -----------------------
    model.eval()
    preds_coords = []
    trues_coords = []
    preds_species = []
    trues_species = []
    scaled_mse_total = 0.0
    scaled_ce_total = 0.0
    n_total = 0
    with torch.no_grad():
        for Xb, ylocb, yspb in test_loader:
            Xb = Xb.to(DEVICE); ylocb = ylocb.to(DEVICE)
            pred_loc_s, pred_sp = model(Xb)
            # accumulate scaled losses (for reference)
            scaled_mse_total += mse(pred_loc_s, ylocb).item() * Xb.size(0)
            if celoss is not None:
                scaled_ce_total += celoss(pred_sp, yspb.to(DEVICE)).item() * Xb.size(0)
            n_total += Xb.size(0)

            # inverse transform to original coords (meters will be computed from degrees)
            pred_loc_np = pred_loc_s.cpu().numpy()
            true_loc_np = ylocb.cpu().numpy()
            pred_unscaled = targ_scaler.inverse_transform(pred_loc_np)   # (batch,2) lat, lon
            true_unscaled = targ_scaler.inverse_transform(true_loc_np)

            preds_coords.append(pred_unscaled)
            trues_coords.append(true_unscaled)

            if pred_sp is not None:
                pred_sp_np = pred_sp.argmax(dim=1).cpu().numpy()
                preds_species.append(pred_sp_np)
                trues_species.append(yspb.numpy())

    if n_total == 0:
        print("WARNING: test set is empty; skipping test evaluation.")
    else:
        scaled_test_mse = scaled_mse_total / n_total
        scaled_test_ce = (scaled_ce_total / n_total) if celoss is not None else None
        preds_coords = np.vstack(preds_coords)
        trues_coords = np.vstack(trues_coords)

        # haversine expects lat1, lon1, lat2, lon2 in degrees and returns meters
        haversine_m = haversine_np(trues_coords[:,0], trues_coords[:,1], preds_coords[:,0], preds_coords[:,1])

        mean_m = float(np.mean(haversine_m))
        median_m = float(np.median(haversine_m))
        rmse_m = float(np.sqrt(np.mean(haversine_m**2)))
        p25, p75, p95 = [float(np.percentile(haversine_m, q)) for q in (25, 75, 95)]

        print("\n=== TEST SET EVALUATION ===")
        print(f"Test samples: {n_total}")
        print(f"Scaled Test MSE (on transformed coords): {scaled_test_mse:.6f}")
        if scaled_test_ce is not None:
            print(f"Scaled Test CE (classification): {scaled_test_ce:.6f}")

        print("\nLocation errors (Haversine, meters):")
        print(f"Mean: {mean_m:.2f} m")
        print(f"Median: {median_m:.2f} m")
        print(f"RMSE: {rmse_m:.2f} m")
        print(f"25th/75th/95th percentiles: {p25:.2f} / {p75:.2f} / {p95:.2f} m")

    # done
if __name__ == "__main__":
    main()
