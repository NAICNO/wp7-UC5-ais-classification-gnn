# Graph construction from AIS time-series
# Each time-series (N, 3, 12) becomes a DGL graph with 12 nodes

import dgl
import torch
import numpy as np

# Example: single AIS sample — 3 features x 12 time steps
sample = np.random.randn(3, 12).astype(np.float32)

# Transpose to (12, 3) — one feature vector per node
node_features = torch.tensor(sample.T, dtype=torch.float32)

# Chain graph: 0→1→2→...→11
edges = [(i, i + 1) for i in range(11)]
graph = dgl.graph(edges)

# Assign features and add self-loops
graph.ndata["attr"] = node_features
graph = dgl.add_self_loop(graph)

print(f"Nodes: {graph.num_nodes()}, Edges: {graph.num_edges()}")
print(f"Features per node: {graph.ndata['attr'].shape}")
