"""Additional tests for utils.py to improve coverage."""
import os
import tempfile

import pytest
try:
    import dgl
except (ImportError, FileNotFoundError, Exception):
    pytest.skip("dgl not available in this environment", allow_module_level=True)
import numpy as np
import pytest
import torch

from graph_classification.utils import (
    get_ais_datasets,
    get_ais_ts_data,
    get_Laplacian_withefeat,
    get_numpy_ds_files,
    plot_metrics,
    sparse_mx_to_torch_sparse_tensor,
)
from graph_classification.ais_timeseries_dataset import (
    get_ais_ts_data as dataset_get_ais_ts_data,
)


class TestSparseConversion:
    """Test scipy sparse to torch sparse conversion."""

    def test_converts_coo_matrix(self):
        import scipy.sparse as sp

        row = np.array([0, 0, 1, 2])
        col = np.array([0, 1, 1, 2])
        data = np.array([1.0, 2.0, 3.0, 4.0])
        sparse_mx = sp.coo_matrix((data, (row, col)), shape=(3, 3))

        result = sparse_mx_to_torch_sparse_tensor(sparse_mx)
        assert result.is_sparse
        assert result.shape == (3, 3)
        dense = result.to_dense()
        assert dense[0, 0] == 1.0
        assert dense[0, 1] == 2.0


class TestGetLaplacianWithEfeat:
    """Test Laplacian computation for graphs with edge features."""

    def test_laplacian_with_edge_features(self):
        g = dgl.graph([(0, 1), (1, 0), (1, 2), (2, 1)])
        g.edata["feat"] = torch.tensor([1.0, 1.0, 1.0, 1.0])
        L = get_Laplacian_withefeat(g, "cpu")
        assert L.shape == (3, 3)


class TestGetAisTsData:
    """Test raw AIS data loading."""

    def test_loads_npy_files_utils(self, tmp_path):
        """Test get_ais_ts_data from utils.py (folder, infile, labelfile)."""
        x = np.random.randn(100, 3, 12).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.float32)
        bidx = np.zeros((5, 100), dtype=int)
        bidx[:, :60] = 1
        bidx[:, 60:80] = 2
        bidx[:, 80:] = 3
        np.save(str(tmp_path / "X_ts12.npy"), x)
        np.save(str(tmp_path / "y_ts12.npy"), y)
        np.save(str(tmp_path / "bidx_ts12.npy"), bidx)

        x_loaded, y_loaded, bidx_loaded = get_ais_ts_data(
            str(tmp_path), "X_ts12.npy", "y_ts12.npy"
        )
        assert x_loaded.shape == (100, 3, 12)
        assert y_loaded.shape == (100,)
        assert bidx_loaded.shape == (5, 100)

    def test_loads_npy_files_dataset(self, tmp_path):
        """Test get_ais_ts_data from ais_timeseries_dataset.py (x_file, y_file)."""
        x = np.random.randn(50, 3, 12).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        x_path = str(tmp_path / "x.npy")
        y_path = str(tmp_path / "y.npy")
        np.save(x_path, x)
        np.save(y_path, y)

        x_loaded, y_loaded = dataset_get_ais_ts_data(x_path, y_path)
        assert x_loaded.shape == (50, 3, 12)
        assert y_loaded.shape == (50,)


class TestGetNumpyDsFiles:
    """Test numpy dataset file discovery and creation."""

    def test_creates_split_files_from_raw(self, tmp_path):
        # Create raw data files
        n_samples = 100
        x = np.random.randn(n_samples, 3, 12).astype(np.float32)
        y = np.random.randint(0, 2, n_samples).astype(np.float32)
        bidx = np.zeros((5, n_samples), dtype=int)
        bidx[:, :60] = 1
        bidx[:, 60:80] = 2
        bidx[:, 80:] = 3

        np.save(str(tmp_path / "X_ts12.npy"), x)
        np.save(str(tmp_path / "y_ts12.npy"), y)
        np.save(str(tmp_path / "bidx_ts12.npy"), bidx)

        result = get_numpy_ds_files(str(tmp_path), k=0)
        assert len(result) == 6
        # All paths should exist after creation
        for path in result:
            assert os.path.exists(path)


class TestGetAisDatasets:
    """Test full dataset loading pipeline."""

    def test_creates_train_val_test(self, tmp_path):
        n_samples = 100
        x = np.random.randn(n_samples, 3, 12).astype(np.float32)
        y = np.random.randint(0, 2, n_samples).astype(np.float32)
        bidx = np.zeros((5, n_samples), dtype=int)
        bidx[:, :60] = 1
        bidx[:, 60:80] = 2
        bidx[:, 80:] = 3

        np.save(str(tmp_path / "X_ts12.npy"), x)
        np.save(str(tmp_path / "y_ts12.npy"), y)
        np.save(str(tmp_path / "bidx_ts12.npy"), bidx)

        train_ds, val_ds, test_ds = get_ais_datasets(str(tmp_path), k=0)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        # Sum should equal total samples
        assert len(train_ds) + len(val_ds) + len(test_ds) == n_samples


class TestPlotMetrics:
    """Test the plotting utility (doesn't verify visual output, just no crash)."""

    def test_plot_single_model(self):
        models = ["GCN"]
        losses = [[0.5, 0.4, 0.3]]
        valid_accs = [[0.8, 0.85, 0.9]]
        test_accs = [[0.78, 0.82, 0.88]]
        # Should not crash
        plot_metrics(models, losses, valid_accs, test_accs)

    def test_plot_multiple_models(self):
        models = ["GCN", "GSG", "GAT"]
        losses = [[0.5, 0.4], [0.6, 0.5], [0.7, 0.6]]
        valid_accs = [[0.8, 0.85], [0.75, 0.80], [0.70, 0.75]]
        test_accs = [[0.78, 0.82], [0.73, 0.78], [0.68, 0.73]]
        plot_metrics(models, losses, valid_accs, test_accs)
