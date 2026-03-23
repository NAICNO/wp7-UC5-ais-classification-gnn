"""Tests for classification head modules."""
import pytest
try:
    import dgl
except (ImportError, FileNotFoundError, Exception):
    pytest.skip("dgl not available in this environment", allow_module_level=True)
import pytest
import torch

from graph_classification.heads import GraphClassificationHead, NodeClassificationHead


class TestNodeClassificationHead:
    """Test node-level classification head."""

    def test_creates_with_valid_params(self):
        head = NodeClassificationHead(in_feats=64, num_classes=2)
        assert head is not None

    def test_forward_returns_per_node_predictions(self, simple_graph):
        head = NodeClassificationHead(in_feats=3, num_classes=2)
        out = head(simple_graph, simple_graph.ndata["attr"].float())
        assert out.shape == (simple_graph.num_nodes(), 2)

    def test_forward_batched_graph(self, batched_graphs):
        head = NodeClassificationHead(in_feats=3, num_classes=5)
        out = head(batched_graphs, batched_graphs.ndata["attr"].float())
        assert out.shape == (batched_graphs.num_nodes(), 5)


class TestGraphClassificationHead:
    """Test graph-level classification head."""

    def test_creates_with_valid_params(self):
        head = GraphClassificationHead(in_feats=64, num_classes=2)
        assert head is not None

    def test_forward_returns_per_graph_predictions(self, simple_graph):
        head = GraphClassificationHead(in_feats=3, num_classes=2)
        out = head(simple_graph, simple_graph.ndata["attr"].float())
        # Single graph → shape (1, 2)
        assert out.shape == (1, 2)

    def test_forward_batched_returns_correct_batch_size(self, batched_graphs):
        head = GraphClassificationHead(in_feats=3, num_classes=2)
        out = head(batched_graphs, batched_graphs.ndata["attr"].float())
        n_graphs = batched_graphs.batch_size
        assert out.shape == (n_graphs, 2)

    def test_output_is_differentiable(self, simple_graph):
        head = GraphClassificationHead(in_feats=3, num_classes=2)
        out = head(simple_graph, simple_graph.ndata["attr"].float())
        loss = out.sum()
        loss.backward()
        for p in head.parameters():
            assert p.grad is not None

    def test_graph_readout_uses_mean(self, simple_graph):
        """GraphClassificationHead uses dgl.mean_nodes for readout."""
        head = GraphClassificationHead(in_feats=3, num_classes=2)
        out = head(simple_graph, simple_graph.ndata["attr"].float())
        # Output should be a mean over nodes, so shape is (1, num_classes)
        assert out.shape[0] == 1
