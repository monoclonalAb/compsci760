import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

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
trajectory_durations = df.groupby("Migratory route codes")["duration_h"].apply(list).tolist() # Added durations as edge-level features

# Training pairs
def build_training_pairs(trajectories, durations, min_len=2):
    pairs = []
    for traj, dur in zip(trajectories, durations):
        if len(traj) < min_len:
            continue
        for i in range(1, len(traj)):
            prefix = traj[:i]
            next_node = traj[i]
            dur_seq = [0.0]
            if i > 1:
                dur_seq.extend(dur[:i-1])  # durations between nodes in prefix
            pairs.append((prefix, next_node, dur_seq))
    return pairs

pairs = build_training_pairs(trajectories, trajectory_durations)
num_nodes = len(unique_coords)
print("Training pairs:", len(pairs), "| Unique nodes:", num_nodes)

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
        prefix, next_node, dur_seq = self.pairs[idx]
        return torch.tensor(prefix, dtype=torch.long), torch.tensor(next_node, dtype=torch.long), torch.tensor(dur_seq, dtype=torch.float)

def collate_fn(batch):
    prefixes, targets, durations = zip(*batch)
    lengths = [len(p) for p in prefixes]
    max_len = max(lengths)
    padded_seq = torch.zeros(len(prefixes), max_len, dtype=torch.long)
    padded_dur = torch.zeros(len(prefixes), max_len, dtype=torch.float)
    for i, (seq, dur) in enumerate(zip(prefixes, durations)):
        padded_seq[i, :len(seq)] = seq
        padded_dur[i, :len(dur)] = dur
    return padded_seq, padded_dur, torch.tensor(lengths), torch.tensor(targets)

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
knn = NearestNeighbors(n_neighbors=5)
knn.fit(coords_scaled)
neighbors = knn.kneighbors_graph(coords_scaled).tocoo()
edge_index = torch.tensor([neighbors.row, neighbors.col], dtype=torch.long)

# Edge weights: all ones
edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)


# ---------------------------
# Step 5. Model definition (Bi-LSTM + Attention)
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

class GCN_BiLSTM_Attn_Temporal(nn.Module):
    def __init__(self, num_nodes, gcn_hidden=64, emb_dim=128, lstm_hidden=128, lstm_layers=1, dropout=0.2, temporal_dim=4, dur_dim=32):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)

        self.gcn1 = GCNConv(2 + emb_dim + temporal_dim, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.duration_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, dur_dim),
        )

        self.lstm = nn.LSTM(emb_dim + dur_dim, lstm_hidden, lstm_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = Attention(2*lstm_hidden)
        self.fc = nn.Linear(2*lstm_hidden, num_nodes)

    def forward(self, x_coords, edge_index, seq, durations, lengths, temporal_features, edge_weight=None):
        node_features = torch.cat([x_coords, self.node_emb.weight, temporal_features], dim=1)
        z = F.relu(self.gcn1(node_features, edge_index, edge_weight=edge_weight))
        z = self.dropout(z)
        z = self.gcn2(z, edge_index, edge_weight=edge_weight)
        z = self.dropout(z)
        seq_emb = z[seq]

        dur_normalized = torch.log1p(durations).unsqueeze(-1)
        dur_emb = self.duration_mlp(dur_normalized)
        combined_emb = torch.cat([seq_emb, dur_emb], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(combined_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        context = self.attention(lstm_out, lengths)
        out = self.fc(context)
        return out

# ---------------------------
# Step 6. Training + evaluation
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GCN_BiLSTM_Attn_Temporal(num_nodes).to(device)
x_coords, edge_index, temporal_features, edge_weight = x_coords.to(device), edge_index.to(device), temporal_features.to(device), edge_weight.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
lambda_mse = 0.1

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for seq, durations, lengths, target in loader:
        seq, durations, lengths, target = seq.to(device), durations.to(device), lengths.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(x_coords, edge_index, seq, durations, lengths, temporal_features, edge_weight=edge_weight)
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
    for seq, durations, lengths, target in loader:
        seq, durations, lengths, target = seq.to(device), durations.to(device), lengths.to(device), target.to(device)
        out = model(x_coords, edge_index, seq, durations, lengths, temporal_features, edge_weight=edge_weight)
        pred_top1 = out.argmax(dim=1)
        correct1 += (pred_top1 == target).sum().item()
        topk = torch.topk(out, k, dim=1).indices
        for i in range(len(target)):
            if target[i] in topk[i]:
                correctk += 1
        mse_total += F.mse_loss(x_coords[pred_top1], x_coords[target], reduction='sum').item()
        total += len(target)
    return correct1/total, correctk/total, mse_total/total

# ---------------------------
# Step 7. Training loop with early stopping
# ---------------------------
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(200):
    loss = train_epoch(model, train_loader)
    top1, top5, mse = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}: Loss {loss:.4f} | Top-1 {top1:.3f} | Top-5 {top5:.3f} | MSE {mse:.4f}")

    if loss < best_val_loss:
        best_val_loss = loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break
