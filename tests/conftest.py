"""Shared fixtures for UC5 graph classification tests."""
import os
import tempfile

os.environ["DGLBACKEND"] = "pytorch"

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

try:
    import dgl
    import torch
    from graph_classification.ais_timeseries_dataset import AISTimeseriesDataset
    DGL_AVAILABLE = True
except (ImportError, Exception):
    DGL_AVAILABLE = False
    dgl = None
    torch = None
    AISTimeseriesDataset = None

dgl_required = pytest.mark.skipif(not DGL_AVAILABLE, reason="dgl not available in this environment")

N_TIMESTEPS = 12
N_FEATURES = 3
N_CLASSES = 2


@pytest.fixture
def make_dataset(tmp_path_factory):
    """Factory fixture that creates AISTimeseriesDataset instances.

    Uses tmp_path_factory so temp directories are cleaned up by pytest.

    Usage in tests:
        def test_something(make_dataset):
            ds, features, labels = make_dataset(n_samples=20, seed=42)
    """
    def _make(n_samples=20, seed=42):
        rng = np.random.RandomState(seed)
        features = rng.randn(n_samples, N_FEATURES, N_TIMESTEPS).astype(np.float32)
        labels = rng.randint(0, N_CLASSES, n_samples).astype(np.float32)

        tmpdir = str(tmp_path_factory.mktemp("ais_data"))
        x_path = os.path.join(tmpdir, "X_ts12.npy")
        y_path = os.path.join(tmpdir, "y_ts12.npy")
        np.save(x_path, features)
        np.save(y_path, labels)

        ds = AISTimeseriesDataset(
            name="test_ds",
            raw_x_file=x_path,
            raw_y_file=y_path,
            save_dir=tmpdir,
        )
        return ds, features, labels

    return _make


@pytest.fixture
def sample_features():
    """Create sample AIS time-series features: (N, 3, 12) array.

    3 features: velocity, distance_to_shore, curvature
    12 time steps per sample
    """
    rng = np.random.RandomState(42)
    return rng.randn(20, N_FEATURES, N_TIMESTEPS).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Binary labels: 0 = non-fishing, 1 = fishing."""
    rng = np.random.RandomState(42)
    return rng.randint(0, N_CLASSES, size=20).astype(np.float32)


@pytest.fixture
def simple_graph():
    """A simple chain graph with 12 nodes (like AIS time-series)."""
    if not DGL_AVAILABLE:
        pytest.skip("dgl not available")
    edges = [(i, i + 1) for i in range(11)]
    g = dgl.graph(edges)
    g.ndata["attr"] = torch.randn(12, 3)
    g = dgl.add_self_loop(g)
    return g


@pytest.fixture
def batched_graphs(simple_graph):
    """A batch of 4 simple graphs."""
    if not DGL_AVAILABLE:
        pytest.skip("dgl not available")
    graphs = []
    for _ in range(4):
        edges = [(i, i + 1) for i in range(11)]
        g = dgl.graph(edges)
        g.ndata["attr"] = torch.randn(12, 3)
        g = dgl.add_self_loop(g)
        graphs.append(g)
    return dgl.batch(graphs)


@pytest.fixture
def device():
    """Return the device to use for testing (CPU)."""
    return "cpu"
