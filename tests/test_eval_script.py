"""Tests for eval_graph_classification_ais module."""
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

from graph_classification.eval_graph_classification_ais import test as eval_test
from graph_classification.train_graph_classification_ais import train


class TestEvalFunction:
    """Test the eval test() function for loading and evaluating saved models."""

    def test_eval_saved_model(self, make_dataset, tmp_path):
        """Train a model, save it, then evaluate it."""
        train_ds, _, _ = make_dataset(30)
        val_ds, _, _ = make_dataset(10, seed=1)
        test_ds, _, _ = make_dataset(10, seed=2)

        model_path = str(tmp_path / "eval_model.pt")
        train(
            device="cpu",
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            model="GCN",
            lr=0.01,
            epochs=3,
            patience=0,
            batch_size=30,
            model_path=model_path,
            pin_memory=False,
            num_workers=0,
        )

        model_name, lr, test_acc, losses, valid_accs, test_accs = eval_test(
            "cpu", test_ds, model_path, batch_size=10
        )
        assert model_name == "GCN"
        assert lr == 0.01
        assert 0.0 <= test_acc <= 1.0
        assert isinstance(losses, list)
        assert isinstance(valid_accs, list)
        assert isinstance(test_accs, list)

    @pytest.mark.parametrize("model_name", ["GCN", "GSG", "GAT"])
    def test_eval_all_model_types(self, make_dataset, tmp_path, model_name):
        """Each model type should save and eval correctly."""
        train_ds, _, _ = make_dataset(30)
        val_ds, _, _ = make_dataset(10, seed=1)
        test_ds, _, _ = make_dataset(10, seed=2)

        model_path = str(tmp_path / f"eval_{model_name}.pt")
        train(
            device="cpu",
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            model=model_name,
            lr=0.01,
            epochs=3,
            patience=0,
            batch_size=30,
            model_path=model_path,
            pin_memory=False,
            num_workers=0,
        )

        loaded_name, lr, test_acc, _, _, _ = eval_test(
            "cpu", test_ds, model_path, batch_size=10
        )
        assert loaded_name == model_name
        assert 0.0 <= test_acc <= 1.0
