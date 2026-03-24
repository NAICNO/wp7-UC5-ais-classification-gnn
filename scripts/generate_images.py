"""Generate result images for README and documentation.

Produces 4 publication-quality figures from synthetic AIS data:
1. Hero image — methodology overview with sample data + graph + results
2. Feature comparison — fishing vs non-fishing feature distributions
3. Model comparison bar chart — accuracy by model and learning rate
4. Training curves — loss and accuracy over epochs

Usage:
    python scripts/generate_images.py
"""

import os
import sys

os.environ["DGLBACKEND"] = "pytorch"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph_classification.models import GCN, GAT, GraphSAGE
from graph_classification.heads import GraphClassificationHead
from graph_classification.ais_timeseries_dataset import AISTimeseriesDataset
from graph_classification.train_graph_classification_ais import train
from graph_classification.utils import create_region_force_model

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "content", "images")


def generate_synthetic_data(n_samples=2000, seed=42):
    """Generate synthetic AIS data mimicking fishing vs non-fishing patterns."""
    rng = np.random.RandomState(seed)
    n_timesteps = 12
    X = np.zeros((n_samples, 3, n_timesteps), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    n_fishing = n_samples // 2

    for i in range(n_fishing):
        X[i, 0] = rng.uniform(2, 6, n_timesteps) + rng.normal(0, 0.5, n_timesteps)
        X[i, 1] = rng.uniform(1, 15, n_timesteps) + rng.normal(0, 1, n_timesteps)
        X[i, 2] = rng.uniform(0.3, 1.0, n_timesteps) + rng.normal(0, 0.1, n_timesteps)
        y[i] = 1.0

    for i in range(n_fishing, n_samples):
        X[i, 0] = rng.uniform(8, 20, n_timesteps) + rng.normal(0, 0.3, n_timesteps)
        X[i, 1] = rng.uniform(10, 50, n_timesteps) + rng.normal(0, 2, n_timesteps)
        X[i, 2] = rng.uniform(0.0, 0.3, n_timesteps) + rng.normal(0, 0.05, n_timesteps)
        y[i] = 0.0

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def make_dataset(X, y, name, tmpdir):
    """Create DGL dataset from numpy arrays."""
    import tempfile
    x_path = os.path.join(tmpdir, f"{name}_X.npy")
    y_path = os.path.join(tmpdir, f"{name}_y.npy")
    np.save(x_path, X)
    np.save(y_path, y)
    return AISTimeseriesDataset(name=name, raw_x_file=x_path, raw_y_file=y_path, save_dir=tmpdir)


def train_all_models(train_ds, val_ds, test_ds):
    """Train all model configurations and return results."""
    models_list = ["GCN", "GSG", "GAT"]
    learning_rates = [0.01, 0.025]
    results = {}

    for model_name in models_list:
        for lr in learning_rates:
            print(f"  Training {model_name} lr={lr}...")
            best_acc, losses, val_accs, test_accs = train(
                device="cpu", train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                seed=0, model=model_name, lr=lr, epochs=60, patience=15,
                batch_size=600, model_path=None, pin_memory=False, num_workers=0,
            )
            results[f"{model_name}_lr_{lr}"] = {
                "model": model_name, "learning_rate": lr,
                "losses": losses, "val_accs": val_accs, "test_accs": test_accs,
                "best_test_acc": best_acc,
            }
    return results


def generate_hero_image(X, y, model_res):
    """Generate hero image combining data, graph structure, and results."""
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)

    # Panel 1: Sample time-series
    ax1 = fig.add_subplot(gs[0])
    fishing_idx = np.where(y == 1)[0][:3]
    nonfishing_idx = np.where(y == 0)[0][:3]
    for idx in fishing_idx:
        ax1.plot(range(12), X[idx, 0, :], color="#e74c3c", alpha=0.7, linewidth=2)
    for idx in nonfishing_idx:
        ax1.plot(range(12), X[idx, 0, :], color="#3498db", alpha=0.7, linewidth=2)
    ax1.set_xlabel("Time Step", fontsize=11)
    ax1.set_ylabel("Velocity", fontsize=11)
    ax1.set_title("AIS Time-Series Data", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    ax1.legend(handles=[
        Line2D([0], [0], color="#e74c3c", label="Fishing", linewidth=2),
        Line2D([0], [0], color="#3498db", label="Non-fishing", linewidth=2),
    ], fontsize=10)

    # Panel 2: Graph structure
    ax2 = fig.add_subplot(gs[1])
    import networkx as nx
    G = nx.path_graph(12)
    pos = {}
    for i in range(12):
        angle = 2 * np.pi * i / 12 - np.pi / 2
        pos[i] = (np.cos(angle), np.sin(angle))
    velocities = X[fishing_idx[0], 0, :]
    norm_v = (velocities - velocities.min()) / (velocities.max() - velocities.min() + 1e-8)
    node_colors = plt.cm.RdYlBu_r(norm_v)
    nx.draw(G, pos, ax=ax2, with_labels=True, node_color=node_colors,
            node_size=450, font_size=8, font_weight="bold",
            edge_color="gray", width=2.5)
    ax2.set_title("Graph Representation", fontsize=13, fontweight="bold")

    # Panel 3: Model comparison
    ax3 = fig.add_subplot(gs[2])
    model_names = []
    accuracies = []
    bar_colors = []
    color_map = {"GCN": "#2ecc71", "GSG": "#3498db", "GAT": "#e74c3c"}
    for key in sorted(model_res.keys()):
        data = model_res[key]
        model_names.append(f'{data["model"]}\nlr={data["learning_rate"]}')
        accuracies.append(data["best_test_acc"] * 100)
        bar_colors.append(color_map[data["model"]])
    bars = ax3.bar(model_names, accuracies, color=bar_colors, edgecolor="white", linewidth=1.5)
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax3.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax3.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax3.set_ylim(bottom=max(0, min(accuracies) - 15), top=105)
    ax3.grid(axis="y", alpha=0.3)
    ax3.axhline(y=50, color="gray", linestyle="--", alpha=0.4)

    plt.savefig(os.path.join(OUTPUT_DIR, "gnn_hero.png"), dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("  Saved gnn_hero.png")


def generate_feature_comparison(X, y):
    """Generate feature distribution comparison between classes."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    feature_names = ["Velocity", "Distance to Shore", "Curvature"]
    fishing_mask = y == 1
    nonfishing_mask = y == 0

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        fishing_vals = X[fishing_mask, i, :].flatten()
        nonfishing_vals = X[nonfishing_mask, i, :].flatten()

        bins = np.linspace(
            min(fishing_vals.min(), nonfishing_vals.min()),
            max(fishing_vals.max(), nonfishing_vals.max()),
            40,
        )
        ax.hist(fishing_vals, bins=bins, alpha=0.6, color="#e74c3c",
                label="Fishing", density=True, edgecolor="white")
        ax.hist(nonfishing_vals, bins=bins, alpha=0.6, color="#3498db",
                label="Non-fishing", density=True, edgecolor="white")
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{name} Distribution", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_comparison.png"), dpi=150,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print("  Saved feature_comparison.png")


def generate_training_curves(model_res):
    """Generate training curves for all models."""
    models_list = ["GCN", "GSG", "GAT"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for model_name in models_list:
        for lr in [0.01, 0.025]:
            key = f"{model_name}_lr_{lr}"
            data = model_res[key]
            color = "#2ecc71" if model_name == "GCN" else "#3498db" if model_name == "GSG" else "#e74c3c"
            style = "-" if lr == 0.01 else "--"
            label = f"{model_name} lr={lr}"
            axes[0].plot(data["losses"], color=color, linestyle=style, label=label, alpha=0.8, linewidth=1.5)
            axes[1].plot(data["val_accs"], color=color, linestyle=style, label=label, alpha=0.8, linewidth=1.5)
            axes[2].plot(data["test_accs"], color=color, linestyle=style, label=label, alpha=0.8, linewidth=1.5)

    for ax, title in zip(axes, ["Training Loss", "Validation Accuracy", "Test Accuracy"]):
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Loss", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[2].set_ylabel("Accuracy", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print("  Saved training_curves.png")


def generate_graph_structure(X, y):
    """Generate graph structure visualization."""
    import networkx as nx

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fishing vessel graph
    fishing_idx = np.where(y == 1)[0][0]
    G = nx.path_graph(12)
    pos = {i: (i, 0) for i in range(12)}
    velocities = X[fishing_idx, 0, :]
    norm_v = (velocities - velocities.min()) / (velocities.max() - velocities.min() + 1e-8)

    ax = axes[0]
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=plt.cm.Reds(norm_v * 0.6 + 0.3),
            node_size=500, font_size=9, font_weight="bold", edge_color="gray", width=2.5)
    ax.set_title("Fishing Vessel (colored by velocity)", fontsize=12, fontweight="bold")

    # Non-fishing vessel graph
    nonfishing_idx = np.where(y == 0)[0][0]
    velocities_nf = X[nonfishing_idx, 0, :]
    norm_nf = (velocities_nf - velocities_nf.min()) / (velocities_nf.max() - velocities_nf.min() + 1e-8)

    ax = axes[1]
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=plt.cm.Blues(norm_nf * 0.6 + 0.3),
            node_size=500, font_size=9, font_weight="bold", edge_color="gray", width=2.5)
    ax.set_title("Non-fishing Vessel (colored by velocity)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "graph_structure.png"), dpi=150,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print("  Saved graph_structure.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    print("Generating synthetic AIS data...")
    X, y = generate_synthetic_data(2000)

    print("Training models...")
    import tempfile
    tmpdir = tempfile.mkdtemp()
    train_ds = make_dataset(X[:1200], y[:1200], "train", tmpdir)
    val_ds = make_dataset(X[1200:1600], y[1200:1600], "val", tmpdir)
    test_ds = make_dataset(X[1600:], y[1600:], "test", tmpdir)
    model_res = train_all_models(train_ds, val_ds, test_ds)

    print("\nGenerating images...")
    generate_hero_image(X, y, model_res)
    generate_feature_comparison(X, y)
    generate_training_curves(model_res)
    generate_graph_structure(X, y)

    print(f"\nDone! Images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
