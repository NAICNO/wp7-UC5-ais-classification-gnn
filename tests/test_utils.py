"""Tests for utility functions."""
import pytest
try:
    import dgl
except (ImportError, FileNotFoundError, Exception):
    pytest.skip("dgl not available in this environment", allow_module_level=True)
import numpy as np
import pandas as pd
import pytest
import torch

from graph_classification.utils import (
    GraphLaplacian,
    ais_data_split,
    create_ais_classification_model_df,
    create_region_force_model,
    get_elapsed_time_str,
    get_Laplacian,
    get_Laplacian_noefeat,
    process_one,
    save_numpy_data,
    transform_graph,
)


class TestGraphLaplacian:
    """Test Graph Laplacian computation."""

    def test_symmetric_laplacian_shape(self):
        adj = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
            values=torch.ones(6),
            size=(3, 3),
        )
        L = GraphLaplacian(adj, symmetric=True)
        assert L.shape == (3, 3)

    def test_asymmetric_laplacian_shape(self):
        adj = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
            values=torch.ones(6),
            size=(3, 3),
        )
        L = GraphLaplacian(adj, symmetric=False)
        assert L.shape == (3, 3)

    def test_laplacian_row_sums_asymmetric(self):
        """For asymmetric (random walk) Laplacian, each row should sum to ~1."""
        adj = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
            values=torch.ones(6),
            size=(3, 3),
        )
        L = GraphLaplacian(adj, symmetric=False)
        row_sums = torch.sparse.sum(L, dim=1).to_dense()
        assert torch.allclose(row_sums, torch.ones(3), atol=1e-6)


class TestGetLaplacian:
    """Test Laplacian computation from DGL graphs."""

    def test_laplacian_from_graph_noefeat(self, simple_graph):
        L = get_Laplacian_noefeat(simple_graph, "cpu")
        assert L.shape == (simple_graph.num_nodes(), simple_graph.num_nodes())

    def test_laplacian_dispatch_by_name(self, simple_graph):
        L = get_Laplacian(simple_graph, "AIS", "cpu")
        assert L.shape == (simple_graph.num_nodes(), simple_graph.num_nodes())


class TestTransformGraph:
    """Test graph transformation utility."""

    def test_removes_self_loops(self):
        g = dgl.graph([(0, 1), (1, 0)])
        g = dgl.add_self_loop(g)
        # Verify self-loops exist before transform
        src, dst = g.edges()
        assert (src == dst).any()
        g_transformed = transform_graph(g)
        src2, dst2 = g_transformed.edges()
        assert not (src2 == dst2).any()

    def test_adds_reverse_edges(self):
        g = dgl.graph([(0, 1), (1, 2)])
        g_transformed = transform_graph(g)
        # Original: 0->1, 1->2 → After reverse: 0->1, 1->2, 1->0, 2->1
        assert g_transformed.num_edges() == 4


class TestCreateRegionForceModel:
    """Test model factory function."""

    @pytest.mark.parametrize("model_name,expected_hidden", [
        ("GCN", 64),
        ("GSG", 64),
        ("GAT", 32),
    ])
    def test_creates_model_with_correct_hidden(self, model_name, expected_hidden):
        model, hidden = create_region_force_model("cpu", 3, model_name)
        assert hidden == expected_hidden
        assert isinstance(model, torch.nn.Module)

    def test_raises_on_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_region_force_model("cpu", 3, "UNKNOWN")


class TestProcessOne:
    """Test the forward pass pipeline."""

    def test_process_one_returns_predictions(self, batched_graphs, device, make_dataset):
        from graph_classification.heads import GraphClassificationHead
        from graph_classification.models import GCN

        dim_nfeats = 3
        gclasses = 2
        init_conv = GCN(dim_nfeats, gclasses, depth=2)
        rf_model, hidden = create_region_force_model(device, dim_nfeats, "GCN")
        head0 = GraphClassificationHead(gclasses, gclasses)
        head = GraphClassificationHead(hidden, gclasses)

        ds, _, _ = make_dataset(5)

        pred = process_one(
            device, ds, batched_graphs, head0, head, init_conv, "attr", rf_model, 3, 1.0 / 3
        )
        assert pred.shape == (batched_graphs.batch_size, gclasses)

    def test_process_one_output_is_bounded(self, batched_graphs, device, make_dataset):
        """Output goes through tanh, so should be in [-1, 1]."""
        from graph_classification.heads import GraphClassificationHead
        from graph_classification.models import GCN

        dim_nfeats = 3
        gclasses = 2
        init_conv = GCN(dim_nfeats, gclasses, depth=2)
        rf_model, hidden = create_region_force_model(device, dim_nfeats, "GCN")
        head0 = GraphClassificationHead(gclasses, gclasses)
        head = GraphClassificationHead(hidden, gclasses)

        ds, _, _ = make_dataset(5)

        pred = process_one(
            device, ds, batched_graphs, head0, head, init_conv, "attr", rf_model, 3, 1.0 / 3
        )
        assert pred.min() >= -1.0
        assert pred.max() <= 1.0


class TestGetElapsedTimeStr:
    """Test time formatting utility."""

    def test_seconds_only(self):
        result = get_elapsed_time_str(45.5)
        assert "45.50 seconds" in result

    def test_minutes_and_seconds(self):
        result = get_elapsed_time_str(125.3)
        assert "2 minutes" in result
        assert "5.30 seconds" in result

    def test_zero_seconds(self):
        result = get_elapsed_time_str(0.0)
        assert "0.00 seconds" in result


class TestAisDataSplit:
    """Test data splitting with bootstrap indices."""

    def test_split_with_k_none(self):
        n_samples = 100
        x_data = np.random.randn(n_samples, 3, 12).astype(np.float32)
        y_data = np.random.randint(0, 2, n_samples).astype(np.float32)
        # bidx shape: (n_bootstraps, n_samples) — values 1=train, 2=val, 3=test
        bidx = np.zeros((5, n_samples), dtype=int)
        bidx[:, :60] = 1  # train
        bidx[:, 60:80] = 2  # val
        bidx[:, 80:] = 3  # test

        x_train, y_train, x_val, y_val, x_test, y_test = ais_data_split(
            None, bidx, x_data, y_data
        )
        assert len(x_train) > 0
        assert len(x_val) > 0
        assert len(x_test) > 0

    def test_split_with_k_index(self):
        n_samples = 100
        x_data = np.random.randn(n_samples, 3, 12).astype(np.float32)
        y_data = np.random.randint(0, 2, n_samples).astype(np.float32)
        bidx = np.zeros((5, n_samples), dtype=int)
        bidx[0, :60] = 1
        bidx[0, 60:80] = 2
        bidx[0, 80:] = 3

        x_train, y_train, x_val, y_val, x_test, y_test = ais_data_split(
            0, bidx, x_data, y_data
        )
        assert len(x_train) == 60
        assert len(x_val) == 20
        assert len(x_test) == 20

    def test_split_removes_overlapping_indices(self):
        """Validation and test indices should exclude training indices."""
        n_samples = 50
        x_data = np.random.randn(n_samples, 3, 12).astype(np.float32)
        y_data = np.random.randint(0, 2, n_samples).astype(np.float32)
        bidx = np.ones((1, n_samples), dtype=int)  # all train
        bidx[0, 40:45] = 2  # val
        bidx[0, 45:] = 3  # test
        # Intentionally mark some as both train and val
        bidx[0, 38:42] = 2  # overlap on 38, 39

        x_train, y_train, x_val, y_val, x_test, y_test = ais_data_split(
            0, bidx, x_data, y_data
        )
        # No overlapping samples
        assert len(x_train) + len(x_val) + len(x_test) <= n_samples


class TestSaveNumpyData:
    """Test numpy data saving."""

    def test_save_and_reload(self, tmp_path):
        x = np.random.randn(10, 3, 12).astype(np.float32)
        y = np.random.randint(0, 2, 10).astype(np.float32)
        x_path = str(tmp_path / "x.npy")
        y_path = str(tmp_path / "y.npy")
        save_numpy_data(x, x_path, y, y_path)
        x_loaded = np.load(x_path)
        y_loaded = np.load(y_path)
        assert np.array_equal(x, x_loaded)
        assert np.array_equal(y, y_loaded)


class TestCreateClassificationModelDf:
    """Test results DataFrame creation."""

    def test_creates_pivot_table(self):
        results = {
            "GCN_lr_0.01": {
                "model": "GCN",
                "learning_rate": 0.01,
                "acc": 0.94,
            },
            "GCN_lr_0.025": {
                "model": "GCN",
                "learning_rate": 0.025,
                "acc": 0.93,
            },
        }
        df = create_ais_classification_model_df(
            results, model_key="model", lr_key="learning_rate", acc_key="acc"
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # 1 model
        assert len(df.columns) == 2  # 2 learning rates

    def test_empty_results_returns_empty_df(self):
        df = create_ais_classification_model_df({})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_multiple_models(self):
        results = {
            "GCN_lr_0.01": {"model": "GCN", "learning_rate": 0.01, "acc": 0.94},
            "GSG_lr_0.01": {"model": "GSG", "learning_rate": 0.01, "acc": 0.94},
            "GAT_lr_0.01": {"model": "GAT", "learning_rate": 0.01, "acc": 0.93},
        }
        df = create_ais_classification_model_df(
            results, model_key="model", lr_key="learning_rate", acc_key="acc"
        )
        assert len(df) == 3  # 3 models
