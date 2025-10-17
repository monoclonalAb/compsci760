import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os

OUT_DIR = "./gat_results"

# -------------------------------
# Load your JSON data
# -------------------------------
data = [
  {
    "config": {
      "k": 3,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 2
    },
    "val_top1": 0.4517241379310345
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 4
    },
    "val_top1": 0.4413793103448276
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 8
    },
    "val_top1": 0.4379310344827586
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 2
    },
    "val_top1": 0.2620689655172414
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 4
    },
    "val_top1": 0.29310344827586204
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 8
    },
    "val_top1": 0.3275862068965517
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 2
    },
    "val_top1": 0.4206896551724138
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 4
    },
    "val_top1": 0.46206896551724136
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 8
    },
    "val_top1": 0.4862068965517241
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 2
    },
    "val_top1": 0.296551724137931
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 4
    },
    "val_top1": 0.3310344827586207
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 8
    },
    "val_top1": 0.35517241379310344
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 2
    },
    "val_top1": 0.4379310344827586
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 4
    },
    "val_top1": 0.43103448275862066
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 8
    },
    "val_top1": 0.49310344827586206
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 2
    },
    "val_top1": 0.2620689655172414
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 4
    },
    "val_top1": 0.3310344827586207
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 8
    },
    "val_top1": 0.3758620689655172
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 2
    },
    "val_top1": 0.4
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 4
    },
    "val_top1": 0.4724137931034483
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 8
    },
    "val_top1": 0.46206896551724136
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 2
    },
    "val_top1": 0.296551724137931
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 4
    },
    "val_top1": 0.3448275862068966
  },
  {
    "config": {
      "k": 3,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 8
    },
    "val_top1": 0.35517241379310344
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 2
    },
    "val_top1": 0.3896551724137931
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 4
    },
    "val_top1": 0.3758620689655172
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 8
    },
    "val_top1": 0.43448275862068964
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 2
    },
    "val_top1": 0.2413793103448276
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 4
    },
    "val_top1": 0.31724137931034485
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 8
    },
    "val_top1": 0.3275862068965517
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 2
    },
    "val_top1": 0.41379310344827586
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 4
    },
    "val_top1": 0.45517241379310347
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 8
    },
    "val_top1": 0.4241379310344828
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 2
    },
    "val_top1": 0.29310344827586204
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 4
    },
    "val_top1": 0.3310344827586207
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 8
    },
    "val_top1": 0.3586206896551724
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 2
    },
    "val_top1": 0.3793103448275862
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 4
    },
    "val_top1": 0.3931034482758621
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 8
    },
    "val_top1": 0.39655172413793105
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 2
    },
    "val_top1": 0.2827586206896552
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 4
    },
    "val_top1": 0.3103448275862069
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 8
    },
    "val_top1": 0.3275862068965517
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 2
    },
    "val_top1": 0.4689655172413793
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 4
    },
    "val_top1": 0.47586206896551725
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 8
    },
    "val_top1": 0.43448275862068964
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 2
    },
    "val_top1": 0.30344827586206896
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 4
    },
    "val_top1": 0.3482758620689655
  },
  {
    "config": {
      "k": 5,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 8
    },
    "val_top1": 0.3620689655172414
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 2
    },
    "val_top1": 0.3793103448275862
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 4
    },
    "val_top1": 0.3896551724137931
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 8
    },
    "val_top1": 0.38620689655172413
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 2
    },
    "val_top1": 0.29310344827586204
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 4
    },
    "val_top1": 0.25517241379310346
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 32,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 8
    },
    "val_top1": 0.2896551724137931
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 2
    },
    "val_top1": 0.3586206896551724
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 4
    },
    "val_top1": 0.42758620689655175
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 8
    },
    "val_top1": 0.43448275862068964
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 2
    },
    "val_top1": 0.30344827586206896
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 4
    },
    "val_top1": 0.30689655172413793
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 32,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 8
    },
    "val_top1": 0.30344827586206896
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 2
    },
    "val_top1": 0.4
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 4
    },
    "val_top1": 0.3896551724137931
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.2,
      "heads": 8
    },
    "val_top1": 0.39655172413793105
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 2
    },
    "val_top1": 0.25517241379310346
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 4
    },
    "val_top1": 0.27241379310344827
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 64,
      "emb_dim": 64,
      "dropout": 0.5,
      "heads": 8
    },
    "val_top1": 0.31724137931034485
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 2
    },
    "val_top1": 0.42758620689655175
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 4
    },
    "val_top1": 0.41724137931034483
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.2,
      "heads": 8
    },
    "val_top1": 0.43103448275862066
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 2
    },
    "val_top1": 0.3310344827586207
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 4
    },
    "val_top1": 0.2793103448275862
  },
  {
    "config": {
      "k": 7,
      "gat_hidden": 64,
      "emb_dim": 128,
      "dropout": 0.5,
      "heads": 8
    },
    "val_top1": 0.3482758620689655
  }
]


# -------------------------------
# Group data by k value
# -------------------------------
grouped = defaultdict(list)
for entry in data:
    grouped[entry["config"]["k"]].append(entry)

# -------------------------------
# Plot results grouped by k
# -------------------------------
fig, axes = plt.subplots(1, len(grouped), figsize=(16,6), sharey=True)

for ax, (k, entries) in zip(axes, grouped.items()):
    val_scores = [e["val_top1"] for e in entries]
    labels = [f'g={e["config"]["gat_hidden"]}, e={e["config"]["emb_dim"]}, d={e["config"]["dropout"]}' for e in entries]
    
    ax.plot(range(len(val_scores)), val_scores, color="orange", marker='o')
    
    # Highlight best
    best_idx = max(range(len(val_scores)), key=lambda i: val_scores[i])
    ax.plot(best_idx, val_scores[best_idx], marker='*', color='red', markersize=15, label="Best")
    ax.annotate(f"{val_scores[best_idx]:.3f}", 
                (best_idx, val_scores[best_idx]), 
                textcoords="offset points", xytext=(0,10), ha='center', color='red')
    
    ax.set_xticks(range(len(val_scores)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_title(f"k={k}")
    ax.set_ylabel("Val Top-1 Accuracy")
    ax.legend()

fig.suptitle("Validation Accuracy grouped by k (best highlighted)", fontsize=14)
plt.tight_layout()
out_plot = os.path.join(OUT_DIR, "stgnn_gat_hyperparameters.png")
plt.savefig(out_plot, dpi=150, bbox_inches='tight')