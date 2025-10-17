import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.calibration import LabelEncoder
from src.preprocessing import haversine
import pandas as pd
import numpy as np
import json
import os

OUT_DIR = "./gat_results"

# ---------------------------
# Step 1. Load & preprocess
# ---------------------------
df = pd.read_excel("data/processed_bird_migration.xlsx").dropna(subset=["GPS_xx", "GPS_yy"])
le_species = LabelEncoder()
df["species_label"] = le_species.fit_transform(df["Bird species"])
num_species = len(le_species.classes_)
species_labels = torch.tensor(df["species_label"].values, dtype=torch.long)

# Node mapping
unique_coords = df[["GPS_xx", "GPS_yy"]].drop_duplicates().reset_index(drop=True)
unique_coords["node_id"] = range(len(unique_coords))
coord_to_id = {(r["GPS_xx"], r["GPS_yy"]): r["node_id"] for _, r in unique_coords.iterrows()}
df["node_id"] = df.apply(lambda r: coord_to_id[(r["GPS_xx"], r["GPS_yy"])], axis=1).astype(int)

coords_raw = unique_coords[["GPS_xx", "GPS_yy"]].values  # Store coordinates before scaling for distance calculations

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
start_month_map = df.groupby(["GPS_xx", "GPS_yy"])["Migration start month"].first().to_dict()
end_month_map   = df.groupby(["GPS_xx", "GPS_yy"])["Migration end month"].first().to_dict()

unique_coords["start_month"] = unique_coords.apply(
    lambda r: start_month_map.get((r["GPS_xx"], r["GPS_yy"]), 1), axis=1
)
unique_coords["end_month"] = unique_coords.apply(
    lambda r: end_month_map.get((r["GPS_xx"], r["GPS_yy"]), 1), axis=1
)

# Species labels mapping
species_map = df.groupby(["GPS_xx", "GPS_yy"])["Bird species"].first().to_dict()
unique_coords["species"] = unique_coords.apply(
    lambda r: species_map.get((r["GPS_xx"], r["GPS_yy"]), "Unknown"), axis=1
)
unique_species = unique_coords["species"].unique()
species_to_id = {s: i for i, s in enumerate(unique_species)}
unique_coords["species_id"] = unique_coords["species"].map(species_to_id)
species_labels = torch.tensor(unique_coords["species_id"].values, dtype=torch.long)
num_species = len(unique_species)
print(f"Number of species: {num_species}")

# ---------------------------
# Step 2. Dataset + Dataloader (70-15-15 split)
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

train_pairs, temp_pairs = train_test_split(pairs, test_size=0.30, random_state=42)
val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.50, random_state=42)

print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

train_loader = DataLoader(TrajectoryDataset(train_pairs), batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(TrajectoryDataset(val_pairs), batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(TrajectoryDataset(test_pairs), batch_size=64, shuffle=False, collate_fn=collate_fn)

# ---------------------------
# Step 3. Node features: coords + temporal
# ---------------------------
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(unique_coords[["GPS_xx", "GPS_yy"]].values)
x_coords = torch.tensor(coords_scaled, dtype=torch.float)

# Cyclic encoding for months
unique_coords["start_month_sin"] = np.sin(2*np.pi*unique_coords["start_month"]/12)
unique_coords["start_month_cos"] = np.cos(2*np.pi*unique_coords["start_month"]/12)
unique_coords["end_month_sin"] = np.sin(2*np.pi*unique_coords["end_month"]/12)
unique_coords["end_month_cos"] = np.cos(2*np.pi*unique_coords["end_month"]/12)

temporal_features = torch.tensor(
    unique_coords[["start_month_sin","start_month_cos","end_month_sin","end_month_cos"]].values,
    dtype=torch.float
)

# ---------------------------
# Step 4. Function to build k-NN graph
# ---------------------------
def build_knn_graph(coords_scaled, k):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(coords_scaled)
    neighbors = knn.kneighbors_graph(coords_scaled).tocoo()
    edge_index = torch.tensor([neighbors.row, neighbors.col], dtype=torch.long)
    return edge_index

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
    def __init__(self, num_nodes, num_species, gat_hidden=64, emb_dim=128, lstm_hidden=128,
                 lstm_layers=1, dropout=0.2, temporal_dim=4, heads=4):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        
        # GAT layers with multi-head attention
        self.gat1 = GATConv(2 + emb_dim + temporal_dim, gat_hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(heads * gat_hidden, emb_dim, heads=1, concat=False, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, lstm_hidden, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0, 
                            bidirectional=True)
        
        # Separate attention heads for multi-task learning
        self.attn_node = Attention(2*lstm_hidden)
        self.attn_species = Attention(2*lstm_hidden)
        
        self.fc_node = nn.Linear(2*lstm_hidden, num_nodes)
        self.fc_species = nn.Linear(2*lstm_hidden, num_species)
        
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
        
        # Context vectors with separate attention
        context_node = self.attn_node(lstm_out, lengths)
        context_species = self.attn_species(lstm_out, lengths)
        
        out_node = self.fc_node(context_node)
        out_species = self.fc_species(context_species)
        return out_node, out_species

# ---------------------------
# Step 6. Training + evaluation helpers
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
x_coords, temporal_features = x_coords.to(device), temporal_features.to(device)
criterion = nn.CrossEntropyLoss()
lambda_mse = 0.1

def train_epoch(model, loader, optimizer, edge_index, species_labels):
    model.train()
    total_loss = 0
    for seq, lengths, target in loader:
        seq, lengths, target = seq.to(device), lengths.to(device), target.to(device)

        species_target = species_labels[target].to(device)
        optimizer.zero_grad()
        out_node, out_species = model(x_coords, edge_index, seq, lengths, temporal_features)
        ce_loss_node = criterion(out_node, target)
        ce_loss_species = criterion(out_species, species_target)
        pred_coords = x_coords[out_node.argmax(dim=1)]
        true_coords = x_coords[target]
        mse_loss = F.mse_loss(pred_coords, true_coords)
        loss = ce_loss_node + lambda_mse * mse_loss + 0.4*ce_loss_species
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, edge_index, x_coords, coords_raw, species_labels, device, k=5):
    model.eval()
    total, correct1, correctk, correct_species = 0, 0, 0, 0
    mse_total, haversine_total = 0.0, 0.0
    for seq, lengths, target in loader:
        seq, lengths, target = seq.to(device), lengths.to(device), target.to(device)
        species_target = species_labels[target].to(device)
        out_node, out_species = model(x_coords, edge_index, seq, lengths, temporal_features)
        pred_top1 = out_node.argmax(dim=1)
        correct1 += (pred_top1 == target).sum().item()
        topk = torch.topk(out_node, k, dim=1).indices
        
        # Species classification
        species_pred = out_species.argmax(dim=1)
        correct_species += (species_pred == species_target).sum().item()

        for i in range(len(target)):
            if target[i] in topk[i]:
                correctk += 1
        mse_total += F.mse_loss(x_coords[pred_top1], x_coords[target], reduction='sum').item()
        
        # Haversine evaluation
        for i in range(len(target)):
            true_node = target[i].item()
            pred_node = pred_top1[i].item()
            pred_coord = coords_raw[pred_node]
            true_coord = coords_raw[true_node]
            haversine_total += haversine(pred_coord[0], pred_coord[1], true_coord[0], true_coord[1])
        
        total += len(target)
    return correct1/total, correctk/total, correct_species/total, mse_total/total, haversine_total/total

# ---------------------------
# Step 7. Training with validation + hyperparam tuning
# ---------------------------
def train_model(config, max_epochs=50, patience=10):
    k = config["k"]
    gat_hidden = config["gat_hidden"]
    emb_dim = config["emb_dim"]
    dropout = config["dropout"]
    heads = config["heads"]

    print(f"\nTraining with config: {config}")
    edge_index = build_knn_graph(coords_scaled, k)
    edge_index = edge_index.to(device)

    model = GAT_BiLSTM_Attn_Temporal(
        num_nodes,
        num_species,
        gat_hidden=gat_hidden,
        emb_dim=emb_dim,
        dropout=dropout,
        heads=heads
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_top1 = 0.0
    best_model_state = None
    counter = 0

    for epoch in range(max_epochs):
        loss = train_epoch(model, train_loader, optimizer, edge_index, species_labels)
        top1_val, top5_val, species_val, mse_val, haversine_val = evaluate(
            model, val_loader, edge_index, x_coords,
            coords_raw, species_labels, device, k=5
        )

        if top1_val > best_val_top1:
            best_val_top1 = top1_val
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break
                
        # Print progress
        print(f"Epoch {epoch+1:03d} | "
              f"Train Loss: {loss:.4f} | "
              f"Val Top-1: {top1_val:.3f} | "
              f"Val Top-5: {top5_val:.3f} | "
              f"Val Species Acc: {species_val:.3f} | "
              f"Val MSE: {mse_val:.4f} | "
              f"Val Haversine: {haversine_val:.4f}")
    
    return best_val_top1, best_model_state, edge_index

# ---------------------------
# Step 8. Hyperparameter search (grid search + logging)
# ---------------------------
candidate_k = [3, 5, 7]
candidate_gat_hidden = [32, 64]
candidate_emb_dim = [64, 128]
candidate_dropout = [0.2, 0.5]
candidate_heads = [2, 4, 8]  # Number of attention heads in GAT

configs = []
for k in candidate_k:
    for g in candidate_gat_hidden:
        for e in candidate_emb_dim:
            for d in candidate_dropout:
                for h in candidate_heads:
                    configs.append({
                        "k": k, 
                        "gat_hidden": g, 
                        "emb_dim": e, 
                        "dropout": d,
                        "heads": h
                    })

results = {}
log_records = []

for cfg in configs:
    val_top1, model_state, edge_index = train_model(cfg)
    results[tuple(cfg.items())] = (val_top1, model_state, edge_index)
    log_records.append({"config": cfg, "val_top1": val_top1})

# Save logs
pd.DataFrame(log_records).to_csv("hyperparam_results_gat.csv", index=False)
with open("hyperparam_results_gat.json", "w") as f:
    json.dump(log_records, f, indent=2)

# Pick best config
best_cfg = max(results, key=lambda c: results[c][0])
print(f"\nBest config: {dict(best_cfg)}")

# ---------------------------
# Step 9. Final evaluation on test set
# ---------------------------
best_val_top1, best_state, best_edge_index = results[best_cfg]
cfg_dict = dict(best_cfg)

final_model = GAT_BiLSTM_Attn_Temporal(
    num_nodes,
    num_species,
    gat_hidden=cfg_dict["gat_hidden"],
    emb_dim=cfg_dict["emb_dim"],
    dropout=cfg_dict["dropout"],
    heads=cfg_dict["heads"]
).to(device)
final_model.load_state_dict(best_state)

top1_test, top5_test, species_test, mse_test, haversine_test = evaluate(
    final_model,
    test_loader,
    best_edge_index,
    x_coords,
    coords_raw,
    species_labels,
    device,
    k=5
)

print(f"\nTest Results with {cfg_dict}: "
      f"Top-1 {top1_test:.3f} | "
      f"Top-5 {top5_test:.3f} | "
      f"Species Acc {species_test:.3f} | "
      f"MSE {mse_test:.4f} | "
      f"Haversine {haversine_test:.4f} km")

# Save final model
torch.save({
    'model_state_dict': best_state,
    'config': cfg_dict,
    'test_metrics': {
        'top1': top1_test,
        'top5': top5_test,
        'species_acc': species_test,
        'mse': mse_test,
        'haversine': haversine_test
    }
}, os.path.join(OUT_DIR, "best_model_gat_final.pt"))

print("\nModel saved to best_model_gat_final.pt")