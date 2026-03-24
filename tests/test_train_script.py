"""Tests for the train_graph_classification_ais module's train function."""
import sys
_dgl_skip_guard = None
try:
    import dgl as _dgl_skip_guard
except (ImportError, FileNotFoundError, Exception):
    import pytest
    pytest.skip("dgl not available in this environment", allow_module_level=True)
import os

import pytest
import torch

from graph_classification.train_graph_classification_ais import train


class TestTrainFunction:
    """Test the full train() function from train_graph_classification_ais."""

    def test_train_gcn_basic(self, make_dataset):
        ds, _, _ = make_dataset(50)
        val_ds, _, _ = make_dataset(20, seed=1)
        test_ds, _, _ = make_dataset(20, seed=2)

        best_acc, losses, valid_accs, test_accs = train(
            device="cpu",
            train_ds=ds,
            val_ds=val_ds,
            test_ds=test_ds,
            seed=0,
            model="GCN",
            lr=0.01,
            epochs=5,
            patience=0,
            batch_size=50,
            model_path=None,
            pin_memory=False,
            num_workers=0,
        )
        assert 0.0 <= best_acc <= 1.0
        assert len(losses) == 5
        assert len(valid_accs) == 5
        assert len(test_accs) == 5

    def test_train_gsg(self, make_dataset):
        ds, _, _ = make_dataset(30)
        val_ds, _, _ = make_dataset(10, seed=1)
        test_ds, _, _ = make_dataset(10, seed=2)

        best_acc, losses, valid_accs, test_accs = train(
            device="cpu",
            train_ds=ds,
            val_ds=val_ds,
            test_ds=test_ds,
            model="GSG",
            lr=0.01,
            epochs=3,
            patience=0,
            batch_size=30,
            model_path=None,
            pin_memory=False,
            num_workers=0,
        )
        assert 0.0 <= best_acc <= 1.0
        assert len(losses) == 3

    def test_train_gat(self, make_dataset):
        ds, _, _ = make_dataset(30)
        val_ds, _, _ = make_dataset(10, seed=1)
        test_ds, _, _ = make_dataset(10, seed=2)

        best_acc, losses, valid_accs, test_accs = train(
            device="cpu",
            train_ds=ds,
            val_ds=val_ds,
            test_ds=test_ds,
            model="GAT",
            lr=0.01,
            epochs=3,
            patience=0,
            batch_size=30,
            model_path=None,
            pin_memory=False,
            num_workers=0,
        )
        assert 0.0 <= best_acc <= 1.0
        assert len(losses) == 3

    def test_train_with_early_stopping(self, make_dataset):
        ds, _, _ = make_dataset(30)
        val_ds, _, _ = make_dataset(10, seed=1)
        test_ds, _, _ = make_dataset(10, seed=2)

        best_acc, losses, valid_accs, test_accs = train(
            device="cpu",
            train_ds=ds,
            val_ds=val_ds,
            test_ds=test_ds,
            model="GCN",
            lr=0.01,
            epochs=100,
            patience=3,
            batch_size=30,
            model_path=None,
            pin_memory=False,
            num_workers=0,
        )
        assert len(losses) < 100

    def test_train_with_model_save(self, make_dataset, tmp_path):
        ds, _, _ = make_dataset(30)
        val_ds, _, _ = make_dataset(10, seed=1)
        test_ds, _, _ = make_dataset(10, seed=2)

        model_path = str(tmp_path / "test_model.pt")
        train(
            device="cpu",
            train_ds=ds,
            val_ds=val_ds,
            test_ds=test_ds,
            model="GCN",
            lr=0.01,
            epochs=5,
            patience=0,
            batch_size=30,
            model_path=model_path,
            pin_memory=False,
            num_workers=0,
        )
        assert os.path.exists(model_path)

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        assert checkpoint["model_name"] == "GCN"
        assert checkpoint["lr"] == 0.01
        assert "init_conv" in checkpoint
        assert "rf_model" in checkpoint
