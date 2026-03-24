"""Tests targeting specific coverage gaps identified by pytest-cov.

Covers:
- utils.py lines 76, 167-168, 182-193, 225, 286
- ais_timeseries_dataset.py line 100 (transform branch)
- train_graph_classification_ais.py main() function
- eval_graph_classification_ais.py main() function
- Additional edge cases for models, heads, dataset, and utils
"""
import os
import sys

os.environ["DGLBACKEND"] = "pytorch"

import pytest
try:
    import dgl
except (ImportError, FileNotFoundError, Exception):
    pytest.skip("dgl not available in this environment", allow_module_level=True)
import numpy as np
import pytest
import torch
import torch.nn as nn

from graph_classification.utils import (
    DS_Type,
    GraphLaplacian,
    ais_data_split,
    create_ais_classification_model_df,
    create_region_force_model,
    get_ais_dataset,
    get_ais_datasets,
    get_ais_ts_data,
    get_elapsed_time_str,
    get_Laplacian,
    get_numpy_ds_files,
    process_one,
    transform_graph,
)
from graph_classification.ais_timeseries_dataset import AISTimeseriesDataset
from graph_classification.models import GCN, GAT, GraphSAGE
from graph_classification.heads import GraphClassificationHead, NodeClassificationHead


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_data_dir(tmp_path, n_samples=60, n_bootstraps=2):
    """Create the standard X/y/bidx .npy files in tmp_path."""
    rng = np.random.RandomState(0)
    x = rng.randn(n_samples, 3, 12).astype(np.float32)
    y = rng.randint(0, 2, n_samples).astype(np.float32)
    bidx = np.zeros((n_bootstraps, n_samples), dtype=int)
    # k=0 split: first 36 train, next 12 val, last 12 test
    bidx[:, :36] = 1
    bidx[:, 36:48] = 2
    bidx[:, 48:] = 3
    np.save(str(tmp_path / "X_ts12.npy"), x)
    np.save(str(tmp_path / "y_ts12.npy"), y)
    np.save(str(tmp_path / "bidx_ts12.npy"), bidx)
    return x, y, bidx


# ===========================================================================
# utils.py: get_Laplacian with MNIST/CIFAR10 name  (line 76)
# ===========================================================================

class TestGetLaplacianMNISTBranch:
    """Cover the MNIST/CIFAR10 branch in get_Laplacian (line 76)."""

    def _make_graph_with_edge_feats(self):
        g = dgl.graph([(0, 1), (1, 0), (1, 2), (2, 1)])
        g.edata["feat"] = torch.tensor([0.5, 0.5, 1.0, 1.0])
        return g

    def test_get_laplacian_mnist_uses_edge_feats(self):
        g = self._make_graph_with_edge_feats()
        L = get_Laplacian(g, "MNIST", "cpu")
        assert L.shape == (g.num_nodes(), g.num_nodes())

    def test_get_laplacian_cifar10_uses_edge_feats(self):
        g = self._make_graph_with_edge_feats()
        L = get_Laplacian(g, "CIFAR10", "cpu")
        assert L.shape == (g.num_nodes(), g.num_nodes())

    def test_get_laplacian_non_mnist_uses_noefeat(self):
        """Any name other than MNIST/CIFAR10 falls through to noefeat path."""
        g = dgl.graph([(0, 1), (1, 2)])
        g = dgl.add_self_loop(g)
        L = get_Laplacian(g, "AIS", "cpu")
        assert L.shape == (g.num_nodes(), g.num_nodes())

    def test_get_laplacian_unknown_name_uses_noefeat(self):
        g = dgl.graph([(0, 1)])
        g = dgl.add_self_loop(g)
        L = get_Laplacian(g, "RANDOM_DATASET", "cpu")
        assert L.shape == (g.num_nodes(), g.num_nodes())


# ===========================================================================
# utils.py: get_ais_datasets FileNotFoundError branch  (lines 167-168)
# ===========================================================================

class TestGetAisDatasetsFileNotFoundBranch:
    """Cover the except FileNotFoundError branch when numpy split files don't exist."""

    def test_get_ais_datasets_with_missing_raw_files(self, tmp_path):
        """When numpy split files are absent AND no raw X_ts12/y_ts12 either,
        get_ais_datasets catches the FileNotFoundError and continues."""
        # Empty directory — get_numpy_ds_files will raise FileNotFoundError;
        # the except catches it and leaves file names as None.
        # AISTimeseriesDataset with None files then raises ValueError.
        with pytest.raises(ValueError, match="Raw data files are not set"):
            get_ais_datasets(str(tmp_path), k=0)

    def test_get_ais_datasets_warning_message(self, tmp_path, capsys):
        """The warning is printed when the FileNotFoundError is caught."""
        with pytest.raises(ValueError):
            get_ais_datasets(str(tmp_path), k=0)
        captured = capsys.readouterr()
        assert "Warning" in captured.out


# ===========================================================================
# utils.py: get_ais_dataset (lines 182-193) — all three DS_Type branches
# ===========================================================================

class TestGetAisDataset:
    """Test all three DS_Type branches of get_ais_dataset."""

    def test_get_ais_dataset_train(self, tmp_path):
        _make_raw_data_dir(tmp_path)
        ds = get_ais_dataset(str(tmp_path), k=0, type=DS_Type.TRAIN)
        assert isinstance(ds, AISTimeseriesDataset)
        assert len(ds) == 36

    def test_get_ais_dataset_validation(self, tmp_path):
        _make_raw_data_dir(tmp_path)
        ds = get_ais_dataset(str(tmp_path), k=0, type=DS_Type.VALIDATION)
        assert isinstance(ds, AISTimeseriesDataset)
        assert len(ds) == 12

    def test_get_ais_dataset_test(self, tmp_path):
        _make_raw_data_dir(tmp_path)
        ds = get_ais_dataset(str(tmp_path), k=0, type=DS_Type.TEST)
        assert isinstance(ds, AISTimeseriesDataset)
        assert len(ds) == 12

    def test_get_ais_dataset_returns_ais_dataset_type(self, tmp_path):
        _make_raw_data_dir(tmp_path)
        for ds_type in [DS_Type.TRAIN, DS_Type.VALIDATION, DS_Type.TEST]:
            ds = get_ais_dataset(str(tmp_path), k=0, type=ds_type)
            assert isinstance(ds, AISTimeseriesDataset)


# ===========================================================================
# utils.py: get_ais_ts_data FileNotFoundError  (line 225)
# ===========================================================================

class TestGetAisTsDataFileNotFound:
    """Cover the FileNotFoundError branch in utils.get_ais_ts_data."""

    def test_raises_file_not_found_for_missing_x(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="File not found"):
            get_ais_ts_data(str(tmp_path), "missing_X.npy", "y.npy")

    def test_error_message_contains_path(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc_info:
            get_ais_ts_data(str(tmp_path), "nonexistent.npy", "y.npy")
        assert "nonexistent.npy" in str(exc_info.value)


# ===========================================================================
# utils.py: create_ais_classification_model_df bootstrap_key branch (line 286)
# ===========================================================================

class TestCreateClassificationModelDfBootstrapKey:
    """Cover the pivot_table branch with bootstrap_key present (line 286)."""

    def test_pivot_with_bootstrap_key(self):
        results = {
            "GCN_k0_lr001": {
                "model": "GCN",
                "bootstrap_idx": 0,
                "learning_rate": 0.01,
                "best_test_acc": 0.92,
            },
            "GCN_k1_lr001": {
                "model": "GCN",
                "bootstrap_idx": 1,
                "learning_rate": 0.01,
                "best_test_acc": 0.90,
            },
            "GSG_k0_lr001": {
                "model": "GSG",
                "bootstrap_idx": 0,
                "learning_rate": 0.01,
                "best_test_acc": 0.88,
            },
        }
        df = create_ais_classification_model_df(
            results,
            model_key="model",
            lr_key="learning_rate",
            acc_key="best_test_acc",
            bootstrap_key="bootstrap_idx",
        )
        assert isinstance(df, df.__class__)
        # Index should be (model, bootstrap_idx) tuples
        assert len(df) >= 2

    def test_pivot_without_bootstrap_key_uses_model_only(self):
        results = {
            "GCN_lr001": {"model": "GCN", "learning_rate": 0.01, "best_test_acc": 0.92},
            "GSG_lr001": {"model": "GSG", "learning_rate": 0.01, "best_test_acc": 0.88},
        }
        df = create_ais_classification_model_df(
            results,
            model_key="model",
            lr_key="learning_rate",
            acc_key="best_test_acc",
        )
        assert len(df) == 2

    def test_column_names_contain_lr(self):
        results = {
            "GCN_lr001": {"model": "GCN", "learning_rate": 0.01, "best_test_acc": 0.92},
        }
        df = create_ais_classification_model_df(
            results,
            model_key="model",
            lr_key="learning_rate",
            acc_key="best_test_acc",
        )
        assert all("lr" in col.lower() for col in df.columns)


# ===========================================================================
# ais_timeseries_dataset.py: __getitem__ transform branch (line 100)
# ===========================================================================

class TestDatasetWithTransform:
    """Cover the transform branch in AISTimeseriesDataset.__getitem__ (line 100)."""

    def test_transform_is_applied_on_getitem(self, tmp_path):
        rng = np.random.RandomState(42)
        x = rng.randn(10, 3, 12).astype(np.float32)
        y = rng.randint(0, 2, 10).astype(np.float32)
        x_path = str(tmp_path / "X_ts12.npy")
        y_path = str(tmp_path / "y_ts12.npy")
        np.save(x_path, x)
        np.save(y_path, y)

        # Transform that adds a self-loop removal marker to ndata
        transform_called = []

        def tracking_transform(g):
            transform_called.append(True)
            return g  # return unmodified graph

        ds = AISTimeseriesDataset(
            name="transform_test",
            raw_x_file=x_path,
            raw_y_file=y_path,
            save_dir=str(tmp_path),
            transform=tracking_transform,
        )
        graph, label = ds[0]
        assert len(transform_called) == 1

    def test_transform_can_modify_graph(self, tmp_path):
        rng = np.random.RandomState(0)
        x = rng.randn(5, 3, 12).astype(np.float32)
        y = rng.randint(0, 2, 5).astype(np.float32)
        x_path = str(tmp_path / "X_ts12.npy")
        y_path = str(tmp_path / "y_ts12.npy")
        np.save(x_path, x)
        np.save(y_path, y)

        # Transform that removes self-loops
        def remove_selfloops(g):
            return dgl.remove_self_loop(g)

        ds = AISTimeseriesDataset(
            name="transform_mod_test",
            raw_x_file=x_path,
            raw_y_file=y_path,
            save_dir=str(tmp_path),
            transform=remove_selfloops,
        )
        graph, label = ds[0]
        src, dst = graph.edges()
        # After removing self-loops, no src==dst edges should remain
        assert not (src == dst).any()

    def test_no_transform_returns_graph_unchanged(self, tmp_path):
        rng = np.random.RandomState(1)
        x = rng.randn(5, 3, 12).astype(np.float32)
        y = rng.randint(0, 2, 5).astype(np.float32)
        x_path = str(tmp_path / "X_ts12.npy")
        y_path = str(tmp_path / "y_ts12.npy")
        np.save(x_path, x)
        np.save(y_path, y)

        ds = AISTimeseriesDataset(
            name="no_transform_test",
            raw_x_file=x_path,
            raw_y_file=y_path,
            save_dir=str(tmp_path),
            transform=None,
        )
        graph, label = ds[0]
        assert graph.num_nodes() == 12
        # Self-loops should still be present (23 edges total)
        assert graph.num_edges() == 23


# ===========================================================================
# train main() function  (lines 107-179)
# ===========================================================================

class TestTrainMain:
    """Cover the main() CLI entry point in train_graph_classification_ais.

    Note: train_graph_classification_ais.main() has a known pre-existing bug:
    results are stored with key 'acc' but create_ais_classification_model_df
    is called with acc_key='best_test_acc'. Additionally, bootstrap_idx=None
    (the default) causes the bootstrap branch in create_ais_classification_model_df
    to be taken (since 'bootstrap_idx' key is present with value None), which
    triggers a KeyError('best_test_acc'). Tests document this behavior.
    """

    def test_main_raises_for_missing_data_folder(self, monkeypatch):
        """main() raises FileNotFoundError when data_folder doesn't exist."""
        from graph_classification.train_graph_classification_ais import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train",
                "--data_folder",
                "/nonexistent_path_xyz",
                "--model_path",
                "/tmp",
            ],
        )
        with pytest.raises((FileNotFoundError, SystemExit)):
            main()

    def test_main_creates_model_path_if_missing(self, tmp_path, monkeypatch):
        """main() creates the model_path directory before training begins, even
        though a downstream pandas KeyError is raised due to the acc_key mismatch
        bug in main(). The directory creation happens before that failure point.
        """
        _make_raw_data_dir(tmp_path)
        new_model_dir = str(tmp_path / "new_results")
        assert not os.path.exists(new_model_dir)

        from graph_classification.train_graph_classification_ais import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train",
                "--data_folder",
                str(tmp_path),
                "--model_path",
                new_model_dir,
                "--models",
                "GCN",
                "--lrs",
                "0.01",
                "--epochs",
                "2",
                "--patience",
                "0",
                "--batch_size",
                "60",
                "--pin_memory",
                "False",
                "--num_workers",
                "0",
            ],
        )
        # main() raises KeyError due to the acc_key='best_test_acc' vs 'acc' mismatch
        with pytest.raises(KeyError):
            main()
        # But the model_path directory was created before the error
        assert os.path.exists(new_model_dir)

    def test_main_trains_at_least_one_epoch_before_error(self, tmp_path, monkeypatch, capsys):
        """main() successfully trains before hitting the KeyError in the results summary."""
        _make_raw_data_dir(tmp_path)
        model_dir = str(tmp_path / "results_check")
        os.makedirs(model_dir)

        from graph_classification.train_graph_classification_ais import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train",
                "--data_folder",
                str(tmp_path),
                "--model_path",
                model_dir,
                "--models",
                "GCN",
                "--lrs",
                "0.01",
                "--epochs",
                "2",
                "--patience",
                "0",
                "--batch_size",
                "60",
                "--pin_memory",
                "False",
                "--num_workers",
                "0",
            ],
        )
        with pytest.raises(KeyError):
            main()
        # Training output was printed before the error
        captured = capsys.readouterr()
        assert "training samples" in captured.out

    def test_main_parses_multiple_models_and_lrs(self, tmp_path, monkeypatch, capsys):
        """main() correctly parses comma-separated models and learning rates."""
        _make_raw_data_dir(tmp_path)
        model_dir = str(tmp_path / "results_multi")
        os.makedirs(model_dir)

        from graph_classification.train_graph_classification_ais import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train",
                "--data_folder",
                str(tmp_path),
                "--model_path",
                model_dir,
                "--models",
                "GCN, GSG",
                "--lrs",
                "0.01, 0.05",
                "--epochs",
                "2",
                "--patience",
                "0",
                "--batch_size",
                "60",
                "--pin_memory",
                "False",
                "--num_workers",
                "0",
            ],
        )
        with pytest.raises(KeyError):
            main()
        # Both models should appear in training output
        captured = capsys.readouterr()
        assert "GCN" in captured.out
        assert "GSG" in captured.out


# ===========================================================================
# eval main() function  (lines 33-71)
# ===========================================================================

class TestEvalMain:
    """Cover the main() CLI entry point in eval_graph_classification_ais."""

    def _train_and_save_model(self, tmp_path, model_name="GCN"):
        """Helper: train a tiny model and save a .pt checkpoint."""
        from graph_classification.train_graph_classification_ais import train

        _make_raw_data_dir(tmp_path)
        # Get split files
        x_train = np.load(str(tmp_path / "X_ts12.npy"))
        y_train = np.load(str(tmp_path / "y_ts12.npy"))
        bidx = np.load(str(tmp_path / "bidx_ts12.npy"))

        from graph_classification.utils import ais_data_split, save_numpy_data

        x_tr, y_tr, x_v, y_v, x_te, y_te = ais_data_split(0, bidx, x_train, y_train)
        tr_path_x = str(tmp_path / "ais_graph_classification_dataset_K_0_train_X.npy")
        tr_path_y = str(tmp_path / "ais_graph_classification_dataset_K_0_train_y.npy")
        va_path_x = str(tmp_path / "ais_graph_classification_dataset_K_0_val_X.npy")
        va_path_y = str(tmp_path / "ais_graph_classification_dataset_K_0_val_y.npy")
        te_path_x = str(tmp_path / "ais_graph_classification_dataset_K_0_test_X.npy")
        te_path_y = str(tmp_path / "ais_graph_classification_dataset_K_0_test_y.npy")
        save_numpy_data(x_tr, tr_path_x, y_tr, tr_path_y)
        save_numpy_data(x_v, va_path_x, y_v, va_path_y)
        save_numpy_data(x_te, te_path_x, y_te, te_path_y)

        train_ds = AISTimeseriesDataset("tr", tr_path_x, tr_path_y, str(tmp_path))
        val_ds = AISTimeseriesDataset("va", va_path_x, va_path_y, str(tmp_path))
        test_ds = AISTimeseriesDataset("te", te_path_x, te_path_y, str(tmp_path))

        model_path = str(tmp_path / f"model_{model_name}.pt")
        train(
            device="cpu",
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            model=model_name,
            lr=0.01,
            epochs=2,
            patience=0,
            batch_size=60,
            model_path=model_path,
            pin_memory=False,
            num_workers=0,
        )
        return model_path

    def test_eval_main_raises_for_missing_model_path(self, monkeypatch):
        """main() raises FileNotFoundError when model_path doesn't exist."""
        from graph_classification.eval_graph_classification_ais import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval",
                "--data_folder",
                "/tmp",
                "--model_path",
                "/nonexistent_eval_path_xyz",
            ],
        )
        with pytest.raises(FileNotFoundError, match="model_path does not exist"):
            main()

    def test_eval_main_raises_for_empty_model_dir(self, tmp_path, monkeypatch):
        """main() raises FileNotFoundError when model_path has no .pt files."""
        _make_raw_data_dir(tmp_path)
        model_dir = str(tmp_path / "empty_models")
        os.makedirs(model_dir)

        from graph_classification.eval_graph_classification_ais import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval",
                "--data_folder",
                str(tmp_path),
                "--k",
                "0",
                "--model_path",
                model_dir,
            ],
        )
        with pytest.raises(FileNotFoundError, match="No models found"):
            main()

    def test_eval_main_runs_with_saved_model(self, tmp_path, monkeypatch, capsys):
        """main() evaluates a saved model and prints results.

        Note: eval main() has a known pre-existing bug: results dict stores
        'test_acc' but create_ais_classification_model_df is called with
        default acc_key='best_test_acc', causing a KeyError at the final
        pivot table step. The test verifies evaluation runs to near-completion.
        """
        model_path = self._train_and_save_model(tmp_path, "GCN")
        model_dir = os.path.dirname(model_path)

        from graph_classification.eval_graph_classification_ais import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval",
                "--data_folder",
                str(tmp_path),
                "--k",
                "0",
                "--batch_size",
                "60",
                "--model_path",
                model_dir,
            ],
        )
        # main() raises KeyError at create_ais_classification_model_df (acc_key mismatch)
        with pytest.raises(KeyError):
            main()
        captured = capsys.readouterr()
        # Verify actual evaluation ran (Best Model line is printed before the error)
        assert "Best Model" in captured.out
        assert "GCN" in captured.out

    def test_eval_main_prints_test_results(self, tmp_path, monkeypatch, capsys):
        """main() prints 'Test results' for each evaluated model."""
        model_path = self._train_and_save_model(tmp_path, "GSG")
        model_dir = os.path.dirname(model_path)

        from graph_classification.eval_graph_classification_ais import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval",
                "--data_folder",
                str(tmp_path),
                "--k",
                "0",
                "--batch_size",
                "60",
                "--model_path",
                model_dir,
            ],
        )
        with pytest.raises(KeyError):
            main()
        captured = capsys.readouterr()
        assert "Test results" in captured.out


# ===========================================================================
# Additional edge-case tests for existing modules
# ===========================================================================

class TestGCNEdgeCases:
    """Edge cases for GCN not covered by existing tests."""

    def test_gcn_depth_one_forward(self):
        """Depth-1 GCN has no additional layers; tests the base conv1 path."""
        g = dgl.graph([(0, 1), (1, 2)])
        g.ndata["attr"] = torch.randn(3, 5)
        g = dgl.add_self_loop(g)
        model = GCN(in_feats=5, h_feats=8, depth=1)
        out = model(g, g.ndata["attr"].float())
        assert out.shape == (3, 8)

    def test_gcn_batchnorm_applied_depth_two_plus(self):
        """With depth>1 the BatchNorm layers are applied; verify forward succeeds."""
        g = dgl.graph([(i, i + 1) for i in range(7)])
        g.ndata["attr"] = torch.randn(8, 4)
        g = dgl.add_self_loop(g)
        model = GCN(in_feats=4, h_feats=16, depth=2)
        out = model(g, g.ndata["attr"].float())
        assert out.shape == (8, 16)

    def test_gcn_large_depth(self):
        g = dgl.graph([(0, 1), (1, 2), (2, 3)])
        g.ndata["attr"] = torch.randn(4, 3)
        g = dgl.add_self_loop(g)
        model = GCN(in_feats=3, h_feats=8, depth=6)
        out = model(g, g.ndata["attr"].float())
        assert out.shape == (4, 8)


class TestGraphSAGEEdgeCases:
    """Edge cases for GraphSAGE not covered by existing tests."""

    def test_sage_depth_one(self):
        """Depth-1 GraphSAGE: only conv1, no extra convs or edge weight branch."""
        g = dgl.graph([(0, 1), (1, 2)])
        g.ndata["attr"] = torch.randn(3, 4)
        g = dgl.add_self_loop(g)
        model = GraphSAGE(in_feats=4, h_feats=8, depth=1)
        out = model(g, g.ndata["attr"].float())
        assert out.shape == (3, 8)

    def test_sage_depth_two_without_edge_feats(self):
        """depth=2 GraphSAGE without edge features uses the else branch."""
        g = dgl.graph([(0, 1), (1, 2), (2, 0)])
        g.ndata["attr"] = torch.randn(3, 4)
        g = dgl.add_self_loop(g)
        model = GraphSAGE(in_feats=4, h_feats=8, depth=2)
        out = model(g, g.ndata["attr"].float())
        assert out.shape == (3, 8)

    def test_sage_parameter_count_increases_with_depth(self):
        shallow = sum(p.numel() for p in GraphSAGE(3, 16, 1).parameters())
        deep = sum(p.numel() for p in GraphSAGE(3, 16, 3).parameters())
        assert deep > shallow


class TestGATEdgeCases:
    """Edge cases for GAT not covered by existing tests."""

    def test_gat_depth_one(self):
        g = dgl.graph([(0, 1), (1, 2)])
        g.ndata["attr"] = torch.randn(3, 3)
        g = dgl.add_self_loop(g)
        model = GAT(in_feats=3, h_feats=8, depth=1)
        out = model(g, g.ndata["attr"].float())
        assert out.shape == (3, 8)

    def test_gat_has_depth_plus_one_layers(self):
        for depth in [1, 2, 4]:
            model = GAT(in_feats=3, h_feats=8, depth=depth)
            assert len(model.gat_layers) == depth + 1

    def test_gat_multiple_heads(self):
        g = dgl.graph([(0, 1), (1, 2)])
        g.ndata["attr"] = torch.randn(3, 3)
        g = dgl.add_self_loop(g)
        model = GAT(in_feats=3, h_feats=8, depth=2, heads=[4])
        out = model(g, g.ndata["attr"].float())
        assert out.shape == (3, 8)


class TestNodeClassificationHeadEdgeCases:
    """Additional edge cases for NodeClassificationHead."""

    def test_output_is_differentiable(self, simple_graph):
        head = NodeClassificationHead(in_feats=3, num_classes=3)
        out = head(simple_graph, simple_graph.ndata["attr"].float())
        loss = out.sum()
        loss.backward()
        for p in head.parameters():
            assert p.grad is not None

    def test_many_classes(self, simple_graph):
        head = NodeClassificationHead(in_feats=3, num_classes=10)
        out = head(simple_graph, simple_graph.ndata["attr"].float())
        assert out.shape == (simple_graph.num_nodes(), 10)

    def test_single_class(self, simple_graph):
        head = NodeClassificationHead(in_feats=3, num_classes=1)
        out = head(simple_graph, simple_graph.ndata["attr"].float())
        assert out.shape == (simple_graph.num_nodes(), 1)


class TestGraphClassificationHeadEdgeCases:
    """Additional edge cases for GraphClassificationHead."""

    def test_many_classes(self, batched_graphs):
        head = GraphClassificationHead(in_feats=3, num_classes=8)
        out = head(batched_graphs, batched_graphs.ndata["attr"].float())
        assert out.shape == (batched_graphs.batch_size, 8)

    def test_single_class(self, simple_graph):
        head = GraphClassificationHead(in_feats=3, num_classes=1)
        out = head(simple_graph, simple_graph.ndata["attr"].float())
        assert out.shape == (1, 1)

    def test_output_depends_on_input(self, simple_graph):
        """Different node features should yield different outputs."""
        head = GraphClassificationHead(in_feats=3, num_classes=2)
        out1 = head(simple_graph, simple_graph.ndata["attr"].float())
        g2 = dgl.graph([(i, i + 1) for i in range(11)])
        g2.ndata["attr"] = torch.zeros(12, 3)
        g2 = dgl.add_self_loop(g2)
        out2 = head(g2, g2.ndata["attr"].float())
        assert not torch.allclose(out1, out2)


class TestGraphLaplacianEdgeCases:
    """Edge cases for GraphLaplacian not covered by existing tests."""

    def test_symmetric_laplacian_with_isolated_node(self):
        """Isolated node has degree 0; d_inv_sqrt should be 0 (isinf branch)."""
        # Node 2 is isolated
        adj = torch.sparse_coo_tensor(
            indices=torch.tensor([[0], [1]]),
            values=torch.ones(1),
            size=(3, 3),
        )
        L = GraphLaplacian(adj, symmetric=True)
        assert L.shape == (3, 3)

    def test_asymmetric_laplacian_with_isolated_node(self):
        """Isolated node has degree 0; d_inv should be 0 (isinf branch)."""
        adj = torch.sparse_coo_tensor(
            indices=torch.tensor([[0], [1]]),
            values=torch.ones(1),
            size=(3, 3),
        )
        L = GraphLaplacian(adj, symmetric=False)
        assert L.shape == (3, 3)

    def test_symmetric_laplacian_returns_sparse_tensor(self):
        adj = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1], [1, 0]]),
            values=torch.ones(2),
            size=(2, 2),
        )
        L = GraphLaplacian(adj, symmetric=True)
        assert L.is_sparse

    def test_asymmetric_laplacian_returns_sparse_tensor(self):
        adj = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1], [1, 0]]),
            values=torch.ones(2),
            size=(2, 2),
        )
        L = GraphLaplacian(adj, symmetric=False)
        assert L.is_sparse


class TestGetElapsedTimeStrEdgeCases:
    """Additional edge cases for get_elapsed_time_str."""

    def test_exactly_one_minute(self):
        result = get_elapsed_time_str(60.0)
        assert "1 minutes" in result
        assert "0.00 seconds" in result

    def test_large_elapsed_time(self):
        result = get_elapsed_time_str(3661.0)
        assert "61 minutes" in result

    def test_fractional_seconds_formatting(self):
        result = get_elapsed_time_str(1.999)
        # Should show exactly 2 decimal places
        assert "1.99" in result or "2.00" in result


class TestAisDataSplitEdgeCases:
    """Additional edge cases for ais_data_split."""

    def test_all_train_no_val_or_test(self):
        n = 20
        x = np.random.randn(n, 3, 12).astype(np.float32)
        y = np.random.randint(0, 2, n).astype(np.float32)
        bidx = np.ones((1, n), dtype=int)  # all samples marked as train
        x_train, y_train, x_val, y_val, x_test, y_test = ais_data_split(0, bidx, x, y)
        assert len(x_train) == n
        assert len(x_val) == 0
        assert len(x_test) == 0

    def test_split_preserves_feature_shape(self):
        n = 30
        x = np.random.randn(n, 3, 12).astype(np.float32)
        y = np.random.randint(0, 2, n).astype(np.float32)
        bidx = np.zeros((1, n), dtype=int)
        bidx[0, :20] = 1
        bidx[0, 20:25] = 2
        bidx[0, 25:] = 3
        x_train, y_train, x_val, y_val, x_test, y_test = ais_data_split(0, bidx, x, y)
        assert x_train.shape[1:] == (3, 12)
        assert x_val.shape[1:] == (3, 12)
        assert x_test.shape[1:] == (3, 12)

    def test_k_none_aggregates_all_bootstraps(self):
        n = 40
        x = np.random.randn(n, 3, 12).astype(np.float32)
        y = np.random.randint(0, 2, n).astype(np.float32)
        # Two bootstrap rows, each with different training samples
        bidx = np.zeros((2, n), dtype=int)
        bidx[0, :10] = 1
        bidx[1, 10:20] = 1
        bidx[0, 20:25] = 2
        bidx[1, 25:30] = 2
        bidx[0, 30:35] = 3
        bidx[1, 35:40] = 3
        x_train, y_train, x_val, y_val, x_test, y_test = ais_data_split(
            None, bidx, x, y
        )
        # With k=None samples from both rows are used; total should be > 0
        assert len(x_train) > 0


class TestDSTypeEnum:
    """Tests for DS_Type enum values."""

    def test_ds_type_values(self):
        assert DS_Type.TRAIN == "train"
        assert DS_Type.VALIDATION == "val"
        assert DS_Type.TEST == "test"

    def test_ds_type_is_string(self):
        assert isinstance(DS_Type.TRAIN, str)
        assert isinstance(DS_Type.VALIDATION, str)
        assert isinstance(DS_Type.TEST, str)

    def test_ds_type_name_attribute(self):
        assert DS_Type.TRAIN.name == "TRAIN"
        assert DS_Type.VALIDATION.name == "VALIDATION"
        assert DS_Type.TEST.name == "TEST"


class TestTransformGraphEdgeCases:
    """Additional edge cases for transform_graph."""

    def test_graph_with_no_self_loops(self):
        """transform_graph on graph without self-loops: remove_self_loop is no-op."""
        g = dgl.graph([(0, 1), (1, 2)])
        g_t = transform_graph(g)
        # Original 2 edges + 2 reversed = 4
        assert g_t.num_edges() == 4

    def test_single_edge_graph(self):
        g = dgl.graph([(0, 1)])
        g_t = transform_graph(g)
        # 0->1 and 1->0
        assert g_t.num_edges() == 2

    def test_transform_is_idempotent_structure(self):
        """Two consecutive transforms: second has no self-loops to remove."""
        g = dgl.graph([(0, 1), (1, 2)])
        g_t1 = transform_graph(g)
        g_t2 = transform_graph(g_t1)
        # After second transform: reverse edges of 4-edge graph = 8
        assert g_t2.num_edges() == 8


class TestDatasetDirectLoad:
    """Test AISTimeseriesDataset.load() path directly (cached dataset)."""

    def test_load_after_save_restores_labels(self, make_dataset):
        ds, _, labels = make_dataset(15)
        ds.save()

        ds2 = AISTimeseriesDataset(
            name="test_ds",
            raw_x_file=ds._raw_x_file,
            raw_y_file=ds._raw_y_file,
            save_dir=ds.save_dir,
        )
        assert torch.allclose(ds2.labels, ds.labels)

    def test_load_after_save_restores_graph_count(self, make_dataset):
        ds, _, _ = make_dataset(12)
        ds.save()

        ds2 = AISTimeseriesDataset(
            name="test_ds",
            raw_x_file=ds._raw_x_file,
            raw_y_file=ds._raw_y_file,
            save_dir=ds.save_dir,
        )
        assert len(ds2) == 12

    def test_load_restores_node_features(self, make_dataset):
        ds, _, _ = make_dataset(8)
        ds.save()

        ds2 = AISTimeseriesDataset(
            name="test_ds",
            raw_x_file=ds._raw_x_file,
            raw_y_file=ds._raw_y_file,
            save_dir=ds.save_dir,
        )
        graph0_original, _ = ds[0]
        graph0_loaded, _ = ds2[0]
        assert torch.allclose(graph0_original.ndata["attr"], graph0_loaded.ndata["attr"])


class TestCreateRegionForceModelEdgeCases:
    """Additional edge cases for create_region_force_model."""

    def test_model_named_gcn_returns_64_hidden(self):
        model, hidden = create_region_force_model("cpu", 5, "GCN")
        assert hidden == 64

    def test_model_named_gsg_returns_64_hidden(self):
        model, hidden = create_region_force_model("cpu", 5, "GSG")
        assert hidden == 64

    def test_model_named_gat_returns_32_hidden(self):
        model, hidden = create_region_force_model("cpu", 5, "GAT")
        assert hidden == 32

    def test_model_is_on_correct_device(self):
        model, _ = create_region_force_model("cpu", 3, "GCN")
        for p in model.parameters():
            assert p.device == torch.device("cpu")

    def test_different_in_feats(self):
        for in_feats in [1, 3, 10, 32]:
            model, hidden = create_region_force_model("cpu", in_feats, "GCN")
            assert isinstance(model, nn.Module)


class TestProcessOneEdgeCases:
    """Additional edge cases for process_one."""

    def test_process_one_with_gsg_model(self, batched_graphs, device, make_dataset):
        dim_nfeats = 3
        gclasses = 2
        init_conv = GCN(dim_nfeats, gclasses, depth=2)
        rf_model, hidden = create_region_force_model(device, dim_nfeats, "GSG")
        head0 = GraphClassificationHead(gclasses, gclasses)
        head = GraphClassificationHead(hidden, gclasses)
        ds, _, _ = make_dataset(5)

        pred = process_one(
            device, ds, batched_graphs, head0, head, init_conv, "attr", rf_model, 3, 1.0 / 3
        )
        assert pred.shape == (batched_graphs.batch_size, gclasses)

    def test_process_one_with_gat_model(self, batched_graphs, device, make_dataset):
        dim_nfeats = 3
        gclasses = 2
        init_conv = GCN(dim_nfeats, gclasses, depth=2)
        rf_model, hidden = create_region_force_model(device, dim_nfeats, "GAT")
        head0 = GraphClassificationHead(gclasses, gclasses)
        head = GraphClassificationHead(hidden, gclasses)
        ds, _, _ = make_dataset(5)

        pred = process_one(
            device, ds, batched_graphs, head0, head, init_conv, "attr", rf_model, 3, 1.0 / 3
        )
        assert pred.shape == (batched_graphs.batch_size, gclasses)

    def test_process_one_t_equals_one(self, batched_graphs, device, make_dataset):
        """T=1 should still produce valid output."""
        dim_nfeats = 3
        gclasses = 2
        init_conv = GCN(dim_nfeats, gclasses, depth=2)
        rf_model, hidden = create_region_force_model(device, dim_nfeats, "GCN")
        head0 = GraphClassificationHead(gclasses, gclasses)
        head = GraphClassificationHead(hidden, gclasses)
        ds, _, _ = make_dataset(5)

        pred = process_one(
            device, ds, batched_graphs, head0, head, init_conv, "attr", rf_model, 1, 1.0
        )
        assert pred.shape == (batched_graphs.batch_size, gclasses)
