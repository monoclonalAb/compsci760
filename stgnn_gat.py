import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Hyperparameters

K = 3
GAT_HIDDEN = 64
EMB_DIM = 64
DROPOUT = 0.2
HEADS = 8

OUT_DIR = "./gat_results"

# ---------------------------
# Step 1. Load & preprocess
# ---------------------------
df = pd.read_excel("data/processed_bird_migration.xlsx").dropna(subset=["GPS_xx", "GPS_yy"])

# Node mapping
unique_coords = df[["GPS_xx", "GPS_yy"]].drop_duplicates().reset_index(drop=True)
unique_coords["node_id"] = range(len(unique_coords))
coord_to_id = {(r["GPS_xx"], r["GPS_yy"]): r["node_id"] for _, r in unique_coords.iterrows()}
df["node_id"] = df.apply(lambda r: coord_to_id[(r["GPS_xx"], r["GPS_yy"])], axis=1).astype(int)


# Trajectories
trajectories = df.groupby("Migratory route codes")["node_id"].apply(list).tolist()

# Training pairs
def build_training_pairs(trajectories, min_len=2):
    pairs = []
    for traj in trajectories:
        if len(traj) < min_len:
            continue
        for i in range(1, len(traj)):
            prefix = traj[:i]
            next_node = traj[i]
            pairs.append((prefix, next_node))
    return pairs

pairs = build_training_pairs(trajectories)
num_nodes = len(unique_coords)
print("Training pairs:", len(pairs), "| Unique nodes:", num_nodes)

# Temporal aspect
# Map start and end months from df
start_month_map = df.groupby(["GPS_xx", "GPS_yy"])["Migration start month"].first().to_dict()
end_month_map   = df.groupby(["GPS_xx", "GPS_yy"])["Migration end month"].first().to_dict()

unique_coords["start_month"] = unique_coords.apply(
    lambda r: start_month_map.get((r["GPS_xx"], r["GPS_yy"]), 1), axis=1
)
unique_coords["end_month"] = unique_coords.apply(
    lambda r: end_month_map.get((r["GPS_xx"], r["GPS_yy"]), 1), axis=1
)

# ---------------------------
# Step 2. Dataset + Dataloader
# ---------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        prefix, next_node = self.pairs[idx]
        return torch.tensor(prefix, dtype=torch.long), torch.tensor(next_node, dtype=torch.long)

def collate_fn(batch):
    prefixes, targets = zip(*batch)
    lengths = [len(p) for p in prefixes]
    max_len = max(lengths)
    padded = torch.zeros(len(prefixes), max_len, dtype=torch.long)
    for i, seq in enumerate(prefixes):
        padded[i, :len(seq)] = seq
    return padded, torch.tensor(lengths), torch.tensor(targets)

train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
train_dataset = TrajectoryDataset(train_pairs)
test_dataset = TrajectoryDataset(test_pairs)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# ---------------------------
# Step 3. Node features: coords + temporal
# ---------------------------
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(unique_coords[["GPS_xx", "GPS_yy"]].values)
x_coords = torch.tensor(coords_scaled, dtype=torch.float)

# Cyclic encoding
unique_coords["start_month_sin"] = np.sin(2*np.pi*unique_coords["start_month"]/12)
unique_coords["start_month_cos"] = np.cos(2*np.pi*unique_coords["start_month"]/12)
unique_coords["end_month_sin"] = np.sin(2*np.pi*unique_coords["end_month"]/12)
unique_coords["end_month_cos"] = np.cos(2*np.pi*unique_coords["end_month"]/12)

temporal_features = torch.tensor(
    unique_coords[["start_month_sin","start_month_cos","end_month_sin","end_month_cos"]].values,
    dtype=torch.float
)
# ---------------------------
# Step 4. Build k-NN edges only
# ---------------------------
# k-NN edges
knn = NearestNeighbors(n_neighbors=K)
knn.fit(coords_scaled)
neighbors = knn.kneighbors_graph(coords_scaled).tocoo()
edge_index = torch.tensor([neighbors.row, neighbors.col], dtype=torch.long)

# ---------------------------
# Step 5. Model definition (GAT + Bi-LSTM + Attention)
# ---------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, lstm_outputs, lengths):
        scores = self.attn(lstm_outputs).squeeze(-1)
        mask = torch.arange(lstm_outputs.size(1), device=lengths.device)[None, :] < lengths[:, None]
        scores[~mask] = float('-inf')
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)
        context = (lstm_outputs * attn_weights).sum(dim=1)
        return context

class GAT_BiLSTM_Attn_Temporal(nn.Module):
    def __init__(self, num_nodes, gat_hidden=GAT_HIDDEN, emb_dim=EMB_DIM, lstm_hidden=128, 
                 lstm_layers=1, dropout=0.2, temporal_dim=4, heads=4):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        
        # GAT layers with multi-head attention
        self.gat1 = GATConv(2 + emb_dim + temporal_dim, gat_hidden, heads=heads, dropout=dropout)
        # Second GAT layer: input is heads*gat_hidden, output is emb_dim
        self.gat2 = GATConv(heads * gat_hidden, emb_dim, heads=1, concat=False, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, lstm_hidden, lstm_layers, batch_first=True, 
                           dropout=dropout if lstm_layers > 1 else 0, bidirectional=True)
        self.attention = Attention(2*lstm_hidden)
        self.fc = nn.Linear(2*lstm_hidden, num_nodes)
        
    def forward(self, x_coords, edge_index, seq, lengths, temporal_features):
        # Concatenate node features
        node_features = torch.cat([x_coords, self.node_emb.weight, temporal_features], dim=1)
        
        # GAT layers - attention weights are learned automatically
        z = F.elu(self.gat1(node_features, edge_index))
        z = self.dropout(z)
        z = self.gat2(z, edge_index)
        z = self.dropout(z)
        
        # Sequence embedding
        seq_emb = z[seq]
        packed = nn.utils.rnn.pack_padded_sequence(seq_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Attention over LSTM outputs
        context = self.attention(lstm_out, lengths)
        out = self.fc(context)
        return out

# ---------------------------
# Step 6. Training + evaluation
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GAT_BiLSTM_Attn_Temporal(num_nodes, heads=4).to(device)
x_coords, edge_index, temporal_features = x_coords.to(device), edge_index.to(device), temporal_features.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
lambda_mse = 0.1

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for seq, lengths, target in loader:
        seq, lengths, target = seq.to(device), lengths.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(x_coords, edge_index, seq, lengths, temporal_features)
        ce_loss = criterion(out, target)
        pred_coords = x_coords[out.argmax(dim=1)]
        true_coords = x_coords[target]
        mse_loss = F.mse_loss(pred_coords, true_coords)
        loss = ce_loss + lambda_mse * mse_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, k=5):
    model.eval()
    total, correct1, correctk, mse_total = 0, 0, 0, 0.0
    for seq, lengths, target in loader:
        seq, lengths, target = seq.to(device), lengths.to(device), target.to(device)
        out = model(x_coords, edge_index, seq, lengths, temporal_features)
        pred_top1 = out.argmax(dim=1)
        correct1 += (pred_top1 == target).sum().item()
        topk = torch.topk(out, k, dim=1).indices
        for i in range(len(target)):
            if target[i] in topk[i]:
                correctk += 1
        mse_total += F.mse_loss(x_coords[pred_top1], x_coords[target], reduction='sum').item()
        total += len(target)
    return correct1/total, correctk/total, mse_total/total

# ---------------------------------------
# Helper: haversine distance (in km)
# ---------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    # convert to radians
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    R = 6371.0  # Earth radius in kilometers
    return R * c

# ---------------------------------------
# Track metrics each epoch
# ---------------------------------------
history = {
    "epoch": [],
    "top1": [],
    "top5": [],
    "mse": [],
    "haversine_rmse": []
}

@torch.no_grad()
def evaluate_with_haversine(model, loader, k=5):
    model.eval()
    total = 0
    correct1 = 0
    correctk = 0
    mse_total = 0.0
    haversine_sq_total = 0.0

    for seq, lengths, target in loader:
        seq = seq.to(device)
        lengths = lengths.to(device)
        target = target.to(device)

        out = model(x_coords, edge_index, seq, lengths, temporal_features)
        pred_top1 = out.argmax(dim=1)

        # ---------- Accuracy ----------
        correct1 += (pred_top1 == target).sum().item()

        topk_indices = torch.topk(out, k, dim=1).indices
        correctk += sum(target[i] in topk_indices[i] for i in range(len(target)))

        # ---------- MSE ----------
        pred_coords_scaled = x_coords[pred_top1]
        true_coords_scaled = x_coords[target]
        mse_total += F.mse_loss(pred_coords_scaled, true_coords_scaled, reduction='sum').item()

        # ---------- Haversine RMSE ----------
        # Lookup original GPS coordinates for predicted and target nodes
        pred_coords = unique_coords.iloc[pred_top1.cpu().numpy()][["GPS_xx", "GPS_yy"]].values
        true_coords = unique_coords.iloc[target.cpu().numpy()][["GPS_xx", "GPS_yy"]].values

        pred_tensor = torch.as_tensor(pred_coords, device=device, dtype=torch.float)
        true_tensor = torch.as_tensor(true_coords, device=device, dtype=torch.float)

        haversine_batch = haversine_distance(
            true_tensor[:, 0], true_tensor[:, 1],
            pred_tensor[:, 0], pred_tensor[:, 1]
        )
        haversine_sq_total += torch.sum(haversine_batch ** 2).item()

        total += len(target)

    # ---------- Aggregate ----------
    top1 = correct1 / total
    top5 = correctk / total
    mse = mse_total / total
    haversine_rmse = math.sqrt(haversine_sq_total / total)

    return top1, top5, mse, haversine_rmse

# ---------------------------
# Training loop with metric logging
# ---------------------------
best_val_loss = float('inf')
patience = 10
counter = 0
num_epochs = 200

for epoch in range(num_epochs):
    loss = train_epoch(model, train_loader)
    top1, top5, mse, haversine_rmse = evaluate_with_haversine(model, test_loader)
    
    # log metrics
    history["epoch"].append(epoch + 1)
    history["top1"].append(top1)
    history["top5"].append(top5)
    history["mse"].append(mse)
    history["haversine_rmse"].append(haversine_rmse)

    print(f"epoch {epoch+1:03d}: loss {loss:.4f} | top-1 {top1:.3f} | top-5 {top5:.3f} | "
          f"mse {mse:.4f} | haversine RMSE {haversine_rmse:.3f} km")

    if loss < best_val_loss:
        best_val_loss = loss
        counter = 0
        torch.save(model.state_dict(), "best_model_gat.pt")
    else:
        counter += 1
        if counter >= patience:
            print("early stopping triggered")
            break


# ---------------------------
# Plot Graph 1: Top-1 / Top-5 Accuracy + MSE
# ---------------------------
epochs = history["epoch"]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Accuracy on left y-axis
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy (%)")
ax1.plot(epochs, np.array(history["top1"]) * 100, label="Top-1 Accuracy", color="tab:blue")
ax1.plot(epochs, np.array(history["top5"]) * 100, label="Top-5 Accuracy", color="tab:orange")
ax1.tick_params(axis='y')
ax1.legend(loc="upper left")

# MSE on right y-axis
ax2 = ax1.twinx()
ax2.set_ylabel("MSE")
ax2.plot(epochs, history["mse"], label="MSE", color="tab:green", linestyle="--")
ax2.tick_params(axis='y', labelcolor="tab:red")

fig.tight_layout()
plt.title("Top-1 / Top-5 Accuracy and MSE over Epochs")
out_plot = os.path.join(OUT_DIR, "stgnn_gat_mse_per_iter.png")
plt.savefig(out_plot, dpi=150, bbox_inches='tight')
plt.close()

# ---------------------------
# Plot Graph 2: Haversine RMSE
# ---------------------------
plt.figure(figsize=(10, 6))
plt.plot(epochs, history["haversine_rmse"], marker='o')
plt.xlabel("Epoch")
plt.ylabel("Haversine RMSE (km)")
plt.title("Haversine RMSE over Epochs")
plt.grid(True)
out_plot = os.path.join(OUT_DIR, "stgnn_gat_rmse_per_iter.png")
plt.savefig(out_plot, dpi=150, bbox_inches='tight')
plt.close()