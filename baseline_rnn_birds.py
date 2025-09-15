# baseline_rnn_birds.py
# Baseline LSTM for next-location regression + species classification
# Requires: pandas, numpy, scikit-learn, torch

import os
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
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

# -----------------------
# Helpers
# -----------------------
def load_and_prepare(df):
    # detect columns used in your spreadsheet
    cur_lon = "GPS_xx"
    cur_lat = "GPS_yy"
    next_lon = "next_longitude"
    next_lat = "next_latitude"
    species_col = "Bird species" if "Bird species" in df.columns else None

    # check presence
    for c in (cur_lon, cur_lat, next_lon, next_lat):
        if c not in df.columns:
            raise RuntimeError(f"Required column '{c}' not found in the sheet.")

    # drop missing essential rows
    df = df.dropna(subset=[cur_lon, cur_lat, next_lon, next_lat]).reset_index(drop=True)

    # optional numeric features to add (feel free to extend)
    extra_feats = []
    if "route_progress" in df.columns:
        extra_feats.append("route_progress")
    if "cumulative_distance" in df.columns:
        extra_feats.append("cumulative_distance")

    # Build feature and target arrays
    X_coords = df[[cur_lat, cur_lon] + extra_feats].astype(float).values
    Y_next = df[[next_lat, next_lon]].astype(float).values

    # species encoding (optional)
    if species_col:
        le = LabelEncoder()
        species_labels = le.fit_transform(df[species_col].astype(str).values)
        num_species = len(le.classes_)
    else:
        le = None
        species_labels = None
        num_species = 0

    return X_coords, Y_next, species_labels, le, num_species

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
# Main training flow
# -----------------------
def main():
    df = pandas.read_excel(DATA_PATH)
    X_coords, Y_next, species_labels, le, num_species = load_and_prepare(df)

    # scale features and targets (fit on whole dataset; if strict eval required, fit only on train)
    feat_scaler = StandardScaler().fit(X_coords)
    targ_scaler = StandardScaler().fit(Y_next)
    X_scaled = feat_scaler.transform(X_coords)
    Y_scaled = targ_scaler.transform(Y_next)

    N, feat_dim = X_scaled.shape[0], X_scaled.shape[1]
    # reshape into sequences (SEQ_LEN), for single-step it's (N, 1, feat_dim)
    X_seq = X_scaled.reshape(N, SEQ_LEN, feat_dim)

    # train/val split (stratify by species if available)
    if species_labels is not None:
        X_tr, X_val, ytr_loc, yval_loc, sp_tr, sp_val = train_test_split(
            X_seq, Y_scaled, species_labels, test_size=0.2, random_state=42, stratify=species_labels)
    else:
        X_tr, X_val, ytr_loc, yval_loc = train_test_split(X_seq, Y_scaled, test_size=0.2, random_state=42)
        sp_tr = sp_val = None

    train_ds = TrajStepDataset(X_tr, ytr_loc, sp_tr)
    val_ds = TrajStepDataset(X_val, yval_loc, sp_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

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
    import pandas as pd
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

if __name__ == "__main__":
    main()
