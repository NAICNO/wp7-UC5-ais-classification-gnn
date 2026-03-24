"""Tests for AISTimeseriesDataset."""
import os

import pytest
try:
    import dgl
except (ImportError, FileNotFoundError, Exception):
    pytest.skip("dgl not available in this environment", allow_module_level=True)
import numpy as np
import pytest
import torch

from graph_classification.ais_timeseries_dataset import AISTimeseriesDataset


class TestDatasetCreation:
    """Test dataset construction from numpy arrays."""

    def test_create_from_numpy_basic(self, make_dataset):
        ds, _, _ = make_dataset(20)
        assert len(ds) == 20

    def test_dataset_has_correct_num_features(self, make_dataset):
        ds, _, _ = make_dataset(20)
        assert ds.dim_nfeats == 3

    def test_dataset_has_correct_num_classes(self, make_dataset):
        ds, _, _ = make_dataset(20)
        assert ds.gclasses == 2

    def test_labels_are_float_tensors(self, make_dataset):
        ds, _, _ = make_dataset(20)
        assert ds.labels.dtype == torch.float32

    def test_raises_when_no_raw_files(self):
        with pytest.raises(ValueError, match="Raw data files are not set"):
            AISTimeseriesDataset(
                name="bad_ds", raw_x_file=None, raw_y_file=None, save_dir=None
            )


class TestDatasetGraphStructure:
    """Test the graph structure created from time-series data."""

    def test_each_graph_has_12_nodes(self, make_dataset):
        ds, _, _ = make_dataset(10)
        for i in range(len(ds)):
            graph, _ = ds[i]
            assert graph.num_nodes() == 12

    def test_graphs_have_sequential_edges(self, make_dataset):
        ds, _, _ = make_dataset(5)
        graph, _ = ds[0]
        # 11 sequential edges + 12 self-loops = 23
        assert graph.num_edges() == 23

    def test_graphs_have_self_loops(self, make_dataset):
        ds, _, _ = make_dataset(5)
        graph, _ = ds[0]
        src, dst = graph.edges()
        self_loops = (src == dst).sum().item()
        assert self_loops == 12  # one per node

    def test_node_features_shape(self, make_dataset):
        ds, _, _ = make_dataset(5)
        graph, _ = ds[0]
        assert graph.ndata["attr"].shape == (12, 3)

    def test_node_features_are_transposed(self, make_dataset):
        """Input is (N, 3, 12), each graph should have features (12, 3)."""
        ds, features, _ = make_dataset(5)
        graph, _ = ds[0]
        expected = torch.tensor(features[0].T, dtype=torch.float32)
        assert torch.allclose(graph.ndata["attr"], expected)


class TestDatasetGetItem:
    """Test __getitem__ and __len__."""

    def test_getitem_returns_graph_and_label(self, make_dataset):
        ds, _, _ = make_dataset(10)
        graph, label = ds[0]
        assert isinstance(graph, dgl.DGLGraph)
        assert isinstance(label, torch.Tensor)

    def test_len_matches_input_count(self, make_dataset):
        ds, features, _ = make_dataset(15)
        assert len(ds) == features.shape[0]

    def test_repr_contains_class_name(self, make_dataset):
        ds, _, _ = make_dataset(20)
        r = repr(ds)
        assert "AISTimeseriesDataset" in r
        assert "20" in r


class TestDatasetSaveLoad:
    """Test dataset save/load functionality."""

    def test_save_and_load_roundtrip(self, make_dataset):
        ds, _, _ = make_dataset(20)
        ds.save()

        ds2 = AISTimeseriesDataset(
            name="test_ds",
            raw_x_file=ds._raw_x_file,
            raw_y_file=ds._raw_y_file,
            save_dir=ds.save_dir,
        )
        assert len(ds2) == len(ds)
        assert ds2.dim_nfeats == ds.dim_nfeats
        assert ds2.gclasses == ds.gclasses

    def test_has_cache_after_save(self, make_dataset):
        ds, _, _ = make_dataset(10)
        ds.save()
        assert ds.has_cache()
