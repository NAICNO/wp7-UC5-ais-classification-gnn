# Graph Neural Network Theory

```{objectives}
- Understand why graph structures are a natural fit for time-series classification
- Learn the message passing framework that underpins all three GNN architectures
- Compare GCN, GraphSAGE, and GAT in terms of aggregation and expressiveness
- Understand the full graph classification pipeline from node features to class predictions
```

```{admonition} Why Graphs Instead of Flat Features?
:class: tip

A traditional ML approach would flatten each 12-step trajectory into a 36-dimensional feature vector (12 steps x 3 features) and feed it to a classifier. This works, but it discards **structural information**: the classifier has no notion that feature 1 and feature 4 are adjacent time steps, or that feature 12 and feature 36 are the same time step measured on different axes.

By representing the trajectory as a graph, we explicitly encode the temporal adjacency. The GNN then learns to propagate information along these edges -- a node representing time step 5 can "see" the velocity change between steps 4 and 6 through message passing, enabling the model to learn local temporal patterns without hand-crafting them.
```

## From Time-Series to Graphs

AIS time-series data is naturally sequential: each sample is a sequence of 12 observations over time. We convert this to a graph by:

1. Creating a **node** for each time step (12 nodes)
2. Adding **edges** between consecutive time steps (chain graph: 0-1, 1-2, ..., 10-11)
3. Assigning the 3 features (velocity, distance to shore, curvature) as **node attributes**
4. Adding **self-loops** to ensure every node receives its own features during message passing

```{figure} ../images/graph_structure.png
:alt: Chain graph structure showing 12 nodes connected sequentially with self-loops
:width: 100%

A single AIS trajectory represented as a chain graph. Each node corresponds to one time step and carries a 3-dimensional feature vector. Edges encode temporal adjacency. Self-loops (not shown for clarity) allow each node to retain its own information during message passing.
```

## Why Message Passing Works for Trajectories

Consider a fishing vessel that slows down at time step 5, turns sharply at step 6, then maintains low speed through steps 7-8. In a flat feature vector, these are just numbers at positions 15-24 (assuming 3 features x 12 steps). A standard MLP has no structural bias to connect step 5's velocity with step 6's curvature.

In a graph, steps 5 and 6 are connected by an edge. After one GNN layer, step 6's representation already contains information about step 5's velocity drop. After two layers, step 6 "knows" about steps 4-8. After three layers, every node has a receptive field covering 6 consecutive steps -- enough to capture most fishing maneuvers.

This locality bias is similar to what CNNs provide for images, but adapted to graph-structured data where the neighborhood is defined by edges rather than spatial proximity.

## The Message Passing Framework

All three GNN architectures in this demonstrator follow the **message passing** paradigm. At each layer, every node:

1. **Collects messages** from its neighbors (including itself via self-loops)
2. **Aggregates** these messages using a permutation-invariant function (sum, mean, or attention-weighted sum)
3. **Updates** its representation by applying a learned transformation

After $L$ layers of message passing, each node's representation encodes information from its $L$-hop neighborhood. For our chain graph with 3 GNN layers, each node can aggregate information from up to 3 steps away in either direction.

Formally, a single message passing layer computes:

$$h_v^{(l+1)} = \text{UPDATE}\left(h_v^{(l)},\; \text{AGGREGATE}\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$$

where $\mathcal{N}(v)$ is the set of neighbors of node $v$ (including $v$ itself via the self-loop).

## Three GNN Architectures

### Graph Convolutional Network (GCN)

GCN extends spectral convolutions to graphs. Each layer aggregates features from neighboring nodes using a symmetric normalized Laplacian. The normalization factor $\frac{1}{\sqrt{d_u d_v}}$ ensures that high-degree nodes do not dominate:

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \frac{1}{\sqrt{d_u d_v}} W^{(l)} h_u^{(l)}\right)$$

**Aggregation**: Weighted sum with degree normalization. All neighbors contribute equally after normalization.

**Strengths**: Simple, computationally efficient, well-understood theoretically. Serves as a strong baseline.

**Limitation**: Fixed aggregation weights -- the model cannot learn that some neighbors are more important than others.

### GraphSAGE (GSG)

GraphSAGE learns by **sampling and aggregating** features from a node's local neighborhood. Unlike GCN, it concatenates the node's own features with the aggregated neighborhood features before transformation:

$$h_v^{(l+1)} = \sigma\left(W \cdot \text{MEAN}\left(\{h_v^{(l)}\} \cup \{h_u^{(l)}, \forall u \in \mathcal{N}(v)\}\right)\right)$$

**Aggregation**: Mean pooling over the neighborhood. The MEAN function is permutation-invariant and naturally normalizes for varying neighborhood sizes.

**Strengths**: Scalable to large graphs through neighborhood sampling. Supports **inductive learning** -- it can generalize to unseen graphs without retraining, which is essential for classifying new vessel trajectories. Produces the most consistent results in our experiments.

**Limitation**: Mean aggregation can lose information about the distribution of neighbor features (e.g., high variance vs. low variance neighborhoods produce similar means).

### Graph Attention Network (GAT)

GAT uses **multi-head attention** to learn different importance weights for different neighbors. Each attention head computes:

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} W h_u^{(l)}\right)$$

where the attention coefficients $\alpha_{vu}$ are computed as:

$$\alpha_{vu} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [W h_v \| W h_u]\right)\right)}{\sum_{k \in \mathcal{N}(v)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [W h_v \| W h_k]\right)\right)}$$

Multiple attention heads are concatenated (intermediate layers) or averaged (final layer) to stabilize training.

**Aggregation**: Attention-weighted sum. The model learns which neighbors to focus on.

**Strengths**: Flexible, data-driven aggregation. Can discover that certain time steps are more informative than others.

**Limitation**: More parameters and higher computational cost. The attention mechanism uses dropout (0.5 in our implementation) for regularization, which can make training less stable. In our experiments, GAT slightly underperforms GCN and GraphSAGE, possibly because the chain graph structure is simple enough that uniform aggregation is sufficient.

### Architecture Comparison

| Property | GCN | GraphSAGE | GAT |
|----------|-----|-----------|-----|
| Aggregation | Normalized sum | Mean | Attention-weighted sum |
| Neighbor weighting | Fixed (degree-based) | Equal (mean) | Learned (attention) |
| Hidden dimension | 64 | 64 | 32 |
| Depth | 3 layers | 3 layers | 3 layers |
| Batch normalization | Yes | Yes | No |
| Dropout | No | No | 0.5 (feature + attention) |
| Best test accuracy | 94.4% | **94.4%** | 93.1% |

## Graph Classification Pipeline

For graph-level classification (fishing vs. non-fishing), the node representations must be aggregated into a single graph-level embedding. The full pipeline is:

1. **Initial GCN layer**: A 2-layer GCN processes raw node features into initial embeddings
2. **Region force model**: The selected GNN backbone (GCN, GraphSAGE, or GAT) with 3 layers further refines node representations
3. **Graph readout**: `dgl.mean_nodes()` computes the mean of all node representations, producing a fixed-size graph embedding
4. **Classification head**: A `GraphConv` layer maps the graph embedding to 2 class logits (fishing vs. non-fishing)
5. **Region force integration**: The initial and refined representations are combined over multiple time steps using a tanh-based integration: $u_{t+1} = \tanh(u_t + f \cdot \Delta t)$, where $f$ is the region force output
6. **Cross-entropy loss**: Standard classification loss for training

The readout step is critical -- it converts variable-sized node sets into fixed-size vectors. Mean pooling works well for our fixed-size graphs (always 12 nodes), but sum or max pooling are alternatives for variable-size graphs.

### Why the Region Force Model?

The region force integration is inspired by neural ordinary differential equations (Neural ODEs). Instead of making a single classification decision from node embeddings, the model iteratively refines its prediction over $T=3$ discrete steps:

$$u_{t+1} = \tanh(u_t + f \cdot \Delta t), \quad t = 0, 1, 2$$

where $u_0$ is the initial classification estimate from the GCN layer and $f$ is the "force" from the GNN backbone. This iterative process allows the model to correct initial misclassifications by applying the learned force field multiple times. The tanh activation keeps the output bounded in $[-1, 1]$.

This approach adds computational overhead (3 forward passes through the classification head) but improves classification accuracy by allowing the model to refine its decision boundary iteratively.

## Over-Smoothing and Depth

A known challenge with GNNs is **over-smoothing**: as the number of layers increases, all node representations converge to similar values, losing discriminative power. For our 12-node chain graph:

- **1-2 layers**: Insufficient receptive field; each node only sees immediate neighbors
- **3 layers** (used in this demonstrator): Good balance; each node covers 6-7 steps
- **4+ layers**: Risk of over-smoothing; all nodes start to look the same

Batch normalization (used in GCN and GraphSAGE) helps mitigate over-smoothing by normalizing node representations at each layer, preventing them from collapsing to a single point.

```{keypoints}
- All three architectures follow the message passing paradigm: collect, aggregate, update
- GCN uses degree-normalized aggregation -- simple and effective as a baseline
- GraphSAGE uses mean aggregation with inductive learning -- best consistency across learning rates
- GAT uses learned attention weights -- most flexible but slightly underperforms on this task
- The graph classification pipeline combines GNN backbone, mean readout, and classification head
- Self-loops are essential to ensure nodes retain their own features during message passing
- Three GNN layers allow each node to see up to 3 hops away in the chain graph
- The region force model integrates initial and refined representations over multiple time steps
```
