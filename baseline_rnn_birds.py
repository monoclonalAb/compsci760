# baseline_rnn_birds.py
# Baseline LSTM for next-location regression + species classification
# Requires: pandas, numpy, scikit-learn, torch

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
DATA_PATH = "./data/processed_bird_migration.xlsx"   # change if needed
MODEL_PATH = "./baseline_rnn_birds.pth"
SCALER_PATH = "./baseline_scalers.npz"

SEQ_LEN = 1             # 1 = single-step model (row -> next row); increase if you create sequences
BATCH_SIZE = 128
HIDDEN_SIZE = 32
NUM_LAYERS = 1
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# -----------------------
# Helpers
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
    """
    Validate required columns exist and return useful config values:
      - names of lat/lon/next_lat/next_lon
      - list of extra numeric features to use
      - route column name to group by (if present)
    """
    cur_lon = "GPS_xx"
    cur_lat = "GPS_yy"
    next_lon = "next_longitude"
    next_lat = "next_latitude"

    # ensure required columns exist
    for c in (cur_lon, cur_lat, next_lon, next_lat):
        if c not in df.columns:
            raise RuntimeError(f"Required column '{c}' not found in the sheet. Columns: {df.columns.tolist()}")

    # optional extras
    extra_feats = []
    if "route_progress" in df.columns:
        extra_feats.append("route_progress")
    if "cumulative_distance" in df.columns:
        extra_feats.append("cumulative_distance")

    # choose route column
    if "Migratory route codes" in df.columns:
        route_col = "Migratory route codes"
    elif "ID" in df.columns:
        route_col = "ID"
    else:
        route_col = None

    return cur_lat, cur_lon, next_lat, next_lon, extra_feats, route_col

# -----------------------
# Dataset class
# -----------------------
class TrajStepDataset(Dataset):
    def __init__(self, X_seq, y_loc, y_sp=None):
        self.X = torch.tensor(X_seq, dtype=torch.float32)   # (N, SEQ_LEN, feat_dim)
        self.y_loc = torch.tensor(y_loc, dtype=torch.float32)  # (N, 2)
        self.y_sp = torch.tensor(y_sp, dtype=torch.long) if y_sp is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y_sp is None:
            return self.X[idx], self.y_loc[idx], -1
        return self.X[idx], self.y_loc[idx], self.y_sp[idx]

# -----------------------
# Model: LSTM -> (regression head, optional classification head)
# -----------------------
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
        out, _ = self.lstm(x)           # out: (batch, seq_len, hidden)
        h = out[:, -1, :]               # last timestep
        loc = self.reg_head(h)
        sp_logits = self.cls_head(h) if self.cls_head is not None else None
        return loc, sp_logits

# -----------------------
# Utility: sliding windows per route
# -----------------------
def create_sliding_windows_per_route(df, route_col, lat_col, lon_col, seq_len=8, stride=1, extras=None, sort_col="route_progress"):
    """
    Build sliding windows within each route. Returns X (N, seq_len, feat_dim), Y (N,2), and meta list (route_id, start_idx).
    """
    X_list, Y_list, meta = [], [], []
    for rid, g in df.groupby(route_col):
        # sort by time/order if column exists
        if sort_col in g.columns:
            g = g.sort_values(sort_col)
        else:
            g = g.sort_index()
        coords = g[[lat_col, lon_col]].values  # (L,2)
        extras_arr = g[extras].values if extras else None
        L = coords.shape[0]
        if L < seq_len + 1:
            continue
        for i in range(0, L - seq_len, stride):
            seq_coords = coords[i:i+seq_len]  # (seq_len,2)
            if extras_arr is not None:
                seq_extras = extras_arr[i:i+seq_len]
                seq = np.concatenate([seq_coords, seq_extras], axis=1)
            else:
                seq = seq_coords
            target = coords[i+seq_len]  # next step
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
# Main training flow
# -----------------------
def main():
    np.random.seed(RANDOM_SEED)
    df = pd.read_excel(DATA_PATH)
    lat_col, lon_col, next_lat, next_lon, extra_feats, route_col = load_and_check_columns(df)

    # basic cleaning: drop rows missing essential coordinates
    df = df.dropna(subset=[lat_col, lon_col, next_lat, next_lon]).reset_index(drop=True)

    # If there is no explicit route column, create synthetic grouping by index blocks
    if route_col is None:
        # create pseudo routes by chunking index into blocks of size SEQ_LEN+1
        block_size = max(SEQ_LEN+1, 10)
        df["_route_tmp"] = (df.index // block_size).astype(int)
        route_col = "_route_tmp"

    # Grouped split by route ids (so entire trajectory belongs to one set)
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

    # assertions
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0

    train_df = df[df[route_col].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df[route_col].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df[route_col].isin(test_ids)].reset_index(drop=True)

    print(f"Routes total={n}, train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print("Rows per split:", len(train_df), len(val_df), len(test_df))

    # create sliding windows per split (within-route windows)
    X_train, Y_train, meta_train = create_sliding_windows_per_route(train_df, route_col, lat_col, lon_col, seq_len=SEQ_LEN, extras=extra_feats)
    X_val,   Y_val,   meta_val   = create_sliding_windows_per_route(val_df,   route_col, lat_col, lon_col, seq_len=SEQ_LEN, extras=extra_feats)
    X_test,  Y_test,  meta_test  = create_sliding_windows_per_route(test_df,  route_col, lat_col, lon_col, seq_len=SEQ_LEN, extras=extra_feats)

    # Fallback: if SEQ_LEN windows couldn't be created for any split, fall back to single-row mapping (row-level)
    if X_train is None:
        # Build row-wise features (current -> next) for train/val/test using the df splits
        def rowwise_arrays(ddf):
            Xr = ddf[[lat_col, lon_col] + extra_feats].astype(float).values
            Yr = ddf[[next_lat, next_lon]].astype(float).values
            return Xr, Yr
        X_train, Y_train = rowwise_arrays(train_df)
        X_val,   Y_val   = rowwise_arrays(val_df)
        X_test,  Y_test  = rowwise_arrays(test_df)
        # reshape to (N, 1, feat_dim)
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

    # Fit scalers on TRAIN only (no leakage)
    feat_dim = X_train.shape[2]
    feat_scaler = StandardScaler().fit(X_train.reshape(-1, feat_dim))
    targ_scaler = StandardScaler().fit(Y_train)

    X_train_s = feat_scaler.transform(X_train.reshape(-1, feat_dim)).reshape(X_train.shape)
    X_val_s   = feat_scaler.transform(X_val.reshape(-1, feat_dim)).reshape(X_val.shape)
    X_test_s  = feat_scaler.transform(X_test.reshape(-1, feat_dim)).reshape(X_test.shape)

    Y_train_s = targ_scaler.transform(Y_train)
    Y_val_s   = targ_scaler.transform(Y_val)
    Y_test_s  = targ_scaler.transform(Y_test)

    # Species encoding: fit on full df to avoid missing labels in val/test (alternatively fit on train and handle unseen)
    species_col = "Bird species" if "Bird species" in df.columns else None
    if species_col:
        le = LabelEncoder()
        le.fit(df[species_col].astype(str).values)  # fit on all species present (not numeric leakage)
        # produce species labels per sequence by taking the mode species for that route in the corresponding split
        def build_species_for_meta(meta_list, split_df):
            species_for_seq = []
            # create mapping route -> mode species for this split
            route_to_species = split_df.groupby(route_col)[species_col].agg(lambda s: s.mode().iloc[0] if len(s.mode())>0 else s.iloc[0]).to_dict()
            for rid, _ in meta_list:
                sp = route_to_species.get(rid, None)
                if sp is None:
                    # fallback: use global mode (should be rare)
                    species_for_seq.append(-1)
                else:
                    species_for_seq.append(int(le.transform([str(sp)])[0]))
            return np.array(species_for_seq, dtype=np.int64)

        sp_train = build_species_for_meta(meta_train, train_df)
        sp_val   = build_species_for_meta(meta_val, val_df)
        sp_test  = build_species_for_meta(meta_test, test_df)

        # If any -1 (missing mapping), set to most common class index (safe fallback)
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

    # Build datasets and loaders
    train_ds = TrajStepDataset(X_train_s, Y_train_s, sp_train)
    val_ds = TrajStepDataset(X_val_s, Y_val_s, sp_val)
    test_ds = TrajStepDataset(X_test_s, Y_test_s, sp_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # build model
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

    # training loop
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
        train_ce = (running_ce / tot) if celoss is not None else None
        val_mse, val_ce = evaluate(val_loader)
        print(f"Epoch {epoch:02d} | Train MSE: {train_mse:.6f}" +
              (f", Train CE: {train_ce:.4f}" if train_ce is not None else "") +
              f" | Val MSE: {val_mse:.6f}" +
              (f", Val CE: {val_ce:.4f}" if val_ce is not None else ""))

    # save model + scalers + classes
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

    # quick sample prediction on validation set
    model.eval()
    Xb, ylocb, yspb = next(iter(val_loader))
    with torch.no_grad():
        pred_loc, pred_sp = model(Xb.to(DEVICE))
    pred_loc = pred_loc.cpu().numpy()
    pred_unscaled = targ_scaler.inverse_transform(pred_loc)
    true_unscaled = targ_scaler.inverse_transform(ylocb.numpy())
    preview = pd.DataFrame({
        "pred_lat": pred_unscaled[:,0][:5],
        "pred_lon": pred_unscaled[:,1][:5],
        "true_lat": true_unscaled[:,0][:5],
        "true_lon": true_unscaled[:,1][:5],
    })
    if le is not None:
        preview["true_species"] = le.inverse_transform(yspb.numpy()[:5])
    print("Example preds (first 5):")
    print(preview.to_string(index=False))

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

        # if preds_species:
        #     preds_species = np.concatenate(preds_species)
        #     trues_species = np.concatenate(trues_species)
        #     acc = accuracy_score(trues_species, preds_species)
        #     f1 = f1_score(trues_species, preds_species, average="macro")
        #     print("\nSpecies classification (test):")
        #     print(f"Accuracy: {acc:.4f}")
        #     print(f"Macro-F1: {f1:.4f}")

    # done
if __name__ == "__main__":
    main()
