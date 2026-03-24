"""Tests for GNN models: GCN, GAT, GraphSAGE."""
import pytest
try:
    import dgl
except (ImportError, FileNotFoundError, Exception):
    pytest.skip("dgl not available in this environment", allow_module_level=True)
import pytest
import torch
import torch.nn as nn

from graph_classification.models import GCN, GAT, GraphSAGE


class TestGCNConstruction:
    """Test GCN model construction and properties."""

    def test_gcn_creates_with_valid_params(self):
        model = GCN(in_feats=3, h_feats=64, depth=2)
        assert isinstance(model, nn.Module)

    def test_gcn_has_correct_depth(self):
        model = GCN(in_feats=3, h_feats=64, depth=4)
        assert model.depth == 4
        assert len(model.convs) == 3  # depth - 1 additional layers
        assert len(model.bns) == 3

    def test_gcn_depth_one_has_no_extra_layers(self):
        model = GCN(in_feats=3, h_feats=32, depth=1)
        assert len(model.convs) == 0
        assert len(model.bns) == 0

    def test_gcn_parameter_count_increases_with_hidden(self):
        small = sum(p.numel() for p in GCN(3, 16, 2).parameters())
        large = sum(p.numel() for p in GCN(3, 64, 2).parameters())
        assert large > small


class TestGCNForward:
    """Test GCN forward pass."""

    def test_gcn_forward_returns_correct_shape(self, simple_graph):
        model = GCN(in_feats=3, h_feats=64, depth=2)
        out = model(simple_graph, simple_graph.ndata["attr"].float())
        assert out.shape == (simple_graph.num_nodes(), 64)

    def test_gcn_forward_batched_graph(self, batched_graphs):
        model = GCN(in_feats=3, h_feats=32, depth=3)
        out = model(batched_graphs, batched_graphs.ndata["attr"].float())
        assert out.shape == (batched_graphs.num_nodes(), 32)

    def test_gcn_forward_different_depths(self, simple_graph):
        for depth in [1, 2, 3, 5]:
            model = GCN(in_feats=3, h_feats=16, depth=depth)
            out = model(simple_graph, simple_graph.ndata["attr"].float())
            assert out.shape == (simple_graph.num_nodes(), 16)

    def test_gcn_output_is_differentiable(self, simple_graph):
        model = GCN(in_feats=3, h_feats=32, depth=2)
        out = model(simple_graph, simple_graph.ndata["attr"].float())
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestGraphSAGEConstruction:
    """Test GraphSAGE model construction."""

    def test_sage_creates_with_valid_params(self):
        model = GraphSAGE(in_feats=3, h_feats=64, depth=2)
        assert isinstance(model, nn.Module)

    def test_sage_has_correct_depth(self):
        model = GraphSAGE(in_feats=3, h_feats=64, depth=4)
        assert model.depth == 4
        assert len(model.convs) == 3
        assert len(model.bns) == 3

    def test_sage_uses_mean_aggregation(self):
        model = GraphSAGE(in_feats=3, h_feats=64, depth=2)
        assert model.conv1._aggre_type == "mean"


class TestGraphSAGEForward:
    """Test GraphSAGE forward pass."""

    def test_sage_forward_returns_correct_shape(self, simple_graph):
        model = GraphSAGE(in_feats=3, h_feats=64, depth=2)
        out = model(simple_graph, simple_graph.ndata["attr"].float())
        assert out.shape == (simple_graph.num_nodes(), 64)

    def test_sage_forward_batched_graph(self, batched_graphs):
        model = GraphSAGE(in_feats=3, h_feats=32, depth=3)
        out = model(batched_graphs, batched_graphs.ndata["attr"].float())
        assert out.shape == (batched_graphs.num_nodes(), 32)

    def test_sage_with_edge_features(self):
        """GraphSAGE should use edge weights when edata['feat'] is present."""
        g = dgl.graph([(0, 1), (1, 2), (2, 3)])
        g.ndata["attr"] = torch.randn(4, 3)
        g.edata["feat"] = torch.ones(3)
        g = dgl.add_self_loop(g)
        # Self-loop edges also need features
        n_edges = g.num_edges()
        g.edata["feat"] = torch.ones(n_edges)
        model = GraphSAGE(in_feats=3, h_feats=16, depth=2)
        out = model(g, g.ndata["attr"].float())
        assert out.shape == (4, 16)

    def test_sage_output_is_differentiable(self, simple_graph):
        model = GraphSAGE(in_feats=3, h_feats=32, depth=2)
        out = model(simple_graph, simple_graph.ndata["attr"].float())
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestGATConstruction:
    """Test GAT model construction."""

    def test_gat_creates_with_valid_params(self):
        model = GAT(in_feats=3, h_feats=32, depth=3)
        assert isinstance(model, nn.Module)

    def test_gat_has_correct_number_of_layers(self):
        model = GAT(in_feats=3, h_feats=32, depth=3)
        assert len(model.gat_layers) == 4  # depth + 1

    def test_gat_custom_heads(self):
        model = GAT(in_feats=3, h_feats=32, depth=2, heads=[4])
        assert isinstance(model, nn.Module)


class TestGATForward:
    """Test GAT forward pass."""

    def test_gat_forward_returns_correct_shape(self, simple_graph):
        model = GAT(in_feats=3, h_feats=32, depth=3)
        out = model(simple_graph, simple_graph.ndata["attr"].float())
        assert out.shape == (simple_graph.num_nodes(), 32)

    def test_gat_forward_batched_graph(self, batched_graphs):
        model = GAT(in_feats=3, h_feats=16, depth=2)
        out = model(batched_graphs, batched_graphs.ndata["attr"].float())
        assert out.shape == (batched_graphs.num_nodes(), 16)

    def test_gat_output_is_differentiable(self, simple_graph):
        model = GAT(in_feats=3, h_feats=32, depth=2)
        out = model(simple_graph, simple_graph.ndata["attr"].float())
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestModelComparison:
    """Test that all models can handle the same inputs consistently."""

    @pytest.mark.parametrize("ModelClass,h_feats", [
        (GCN, 64),
        (GraphSAGE, 64),
        (GAT, 32),
    ])
    def test_all_models_accept_same_input(self, simple_graph, ModelClass, h_feats):
        model = ModelClass(in_feats=3, h_feats=h_feats, depth=2)
        out = model(simple_graph, simple_graph.ndata["attr"].float())
        assert out.shape[0] == simple_graph.num_nodes()
        assert out.shape[1] == h_feats
