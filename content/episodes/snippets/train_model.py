# Train a GNN model on AIS graph classification data
import os
os.environ["DGLBACKEND"] = "pytorch"

from graph_classification.utils import get_ais_datasets
from graph_classification.train_graph_classification_ais import train

# Load dataset
train_ds, val_ds, test_ds = get_ais_datasets("data/", k=0)

# Train GraphSAGE (best model)
best_acc, losses, val_accs, test_accs = train(
    device="cuda:0",
    train_ds=train_ds,
    val_ds=val_ds,
    test_ds=test_ds,
    model="GSG",
    lr=0.01,
    epochs=100,
    patience=20,
    batch_size=600,
    model_path="results/best_gsg.pt",
)

print(f"Best test accuracy: {best_acc:.4f}")
