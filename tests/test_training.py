"""Tests for training and evaluation pipeline."""
import sys
_dgl_skip_guard = None
try:
    import dgl as _dgl_skip_guard
except (ImportError, FileNotFoundError, Exception):
    import pytest
    pytest.skip("dgl not available in this environment", allow_module_level=True)
import os

import numpy as np
import pytest
import torch
from dgl.dataloading import GraphDataLoader

from graph_classification.ais_timeseries_dataset import AISTimeseriesDataset
from graph_classification.heads import GraphClassificationHead
from graph_classification.models import GCN, GAT, GraphSAGE
from graph_classification.utils import (
    create_region_force_model,
    get_test_result,
    process_one,
)


class TestTrainingPipeline:
    """Test the core training loop components."""

    def test_dataloader_yields_batched_graphs(self, make_dataset):
        ds, _, _ = make_dataset(50)
        loader = GraphDataLoader(ds, batch_size=10, drop_last=False)
        batched_graph, labels = next(iter(loader))
        assert batched_graph.batch_size == 10
        assert labels.shape == (10,)

    def test_single_training_step(self, make_dataset):
        """Test one forward + backward pass doesn't crash."""
        torch.manual_seed(42)
        ds, _, _ = make_dataset(20)
        loader = GraphDataLoader(ds, batch_size=20, drop_last=False)
        device = "cpu"
        dim_nfeats = 3
        gclasses = 2

        init_conv = GCN(dim_nfeats, gclasses, depth=2)
        rf_model, hidden = create_region_force_model(device, dim_nfeats, "GCN")
        head0 = GraphClassificationHead(gclasses, gclasses)
        head = GraphClassificationHead(hidden, gclasses)

        optimizer = torch.optim.Adam(
            list(rf_model.parameters())
            + list(head.parameters())
            + list(init_conv.parameters())
            + list(head0.parameters()),
            lr=0.01,
        )

        batched_graph, labels = next(iter(loader))
        pred = process_one(
            device, ds, batched_graph, head0, head, init_conv, "attr", rf_model, 3, 1.0 / 3
        )
        loss = torch.nn.functional.cross_entropy(pred, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    @pytest.mark.parametrize("model_name", ["GCN", "GSG", "GAT"])
    def test_training_step_all_models(self, make_dataset, model_name):
        """All model types should complete a training step without error."""
        torch.manual_seed(42)
        ds, _, _ = make_dataset(20)
        loader = GraphDataLoader(ds, batch_size=20, drop_last=False)
        device = "cpu"
        dim_nfeats = 3
        gclasses = 2

        init_conv = GCN(dim_nfeats, gclasses, depth=2)
        rf_model, hidden = create_region_force_model(device, dim_nfeats, model_name)
        head0 = GraphClassificationHead(gclasses, gclasses)
        head = GraphClassificationHead(hidden, gclasses)

        optimizer = torch.optim.Adam(
            list(rf_model.parameters())
            + list(head.parameters())
            + list(init_conv.parameters())
            + list(head0.parameters()),
            lr=0.01,
        )

        batched_graph, labels = next(iter(loader))
        pred = process_one(
            device, ds, batched_graph, head0, head, init_conv, "attr", rf_model, 3, 1.0 / 3
        )
        loss = torch.nn.functional.cross_entropy(pred, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert not torch.isnan(loss)

    def test_loss_decreases_over_epochs(self, make_dataset):
        """Training for a few epochs should reduce loss."""
        torch.manual_seed(42)
        ds, _, _ = make_dataset(50)
        loader = GraphDataLoader(ds, batch_size=50, drop_last=False)
        device = "cpu"
        dim_nfeats = 3
        gclasses = 2

        init_conv = GCN(dim_nfeats, gclasses, depth=2)
        rf_model, hidden = create_region_force_model(device, dim_nfeats, "GCN")
        head0 = GraphClassificationHead(gclasses, gclasses)
        head = GraphClassificationHead(hidden, gclasses)

        optimizer = torch.optim.Adam(
            list(rf_model.parameters())
            + list(head.parameters())
            + list(init_conv.parameters())
            + list(head0.parameters()),
            lr=0.01,
        )

        losses = []
        for epoch in range(10):
            for batched_graph, labels in loader:
                pred = process_one(
                    device, ds, batched_graph, head0, head, init_conv, "attr", rf_model, 3, 1.0 / 3
                )
                loss = torch.nn.functional.cross_entropy(pred, labels.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        # Min loss should be less than first loss (robust to non-monotonic descent)
        assert min(losses[-3:]) < losses[0]


class TestEvaluation:
    """Test evaluation / accuracy computation."""

    def test_get_test_result_returns_accuracy(self, make_dataset):
        ds, _, _ = make_dataset(20)
        loader = GraphDataLoader(ds, batch_size=20, drop_last=False)
        device = "cpu"
        dim_nfeats = 3
        gclasses = 2

        init_conv = GCN(dim_nfeats, gclasses, depth=2)
        rf_model, hidden = create_region_force_model(device, dim_nfeats, "GCN")
        head0 = GraphClassificationHead(gclasses, gclasses)
        head = GraphClassificationHead(hidden, gclasses)

        acc = get_test_result(
            device, loader, ds, head0, head, init_conv, "attr", rf_model, 3, 1.0 / 3
        )
        assert 0.0 <= acc <= 1.0

    def test_accuracy_in_valid_range_after_training(self, make_dataset):
        """After some training, accuracy should be in valid range [0, 1]."""
        torch.manual_seed(42)
        ds, _, _ = make_dataset(100, seed=0)
        loader = GraphDataLoader(ds, batch_size=100, drop_last=False)
        device = "cpu"
        dim_nfeats = 3
        gclasses = 2

        init_conv = GCN(dim_nfeats, gclasses, depth=2)
        rf_model, hidden = create_region_force_model(device, dim_nfeats, "GCN")
        head0 = GraphClassificationHead(gclasses, gclasses)
        head = GraphClassificationHead(hidden, gclasses)

        optimizer = torch.optim.Adam(
            list(rf_model.parameters())
            + list(head.parameters())
            + list(init_conv.parameters())
            + list(head0.parameters()),
            lr=0.01,
        )

        for _ in range(20):
            for batched_graph, labels in loader:
                pred = process_one(
                    device, ds, batched_graph, head0, head, init_conv, "attr", rf_model, 3, 1.0 / 3
                )
                loss = torch.nn.functional.cross_entropy(pred, labels.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        init_conv.eval()
        rf_model.eval()
        head0.eval()
        head.eval()
        with torch.no_grad():
            acc = get_test_result(
                device, loader, ds, head0, head, init_conv, "attr", rf_model, 3, 1.0 / 3
            )
        assert 0.0 <= acc <= 1.0


class TestModelSaveLoad:
    """Test model checkpoint save/load cycle.

    Note: The production code saves full nn.Module objects (not state_dicts),
    requiring weights_only=False on load. This is the existing pattern from
    the upstream NORCE codebase.
    """

    def test_save_and_load_checkpoint(self, tmp_path):
        device = "cpu"
        dim_nfeats = 3
        gclasses = 2

        init_conv = GCN(dim_nfeats, gclasses, depth=2)
        rf_model, hidden = create_region_force_model(device, dim_nfeats, "GCN")
        head0 = GraphClassificationHead(gclasses, gclasses)
        head = GraphClassificationHead(hidden, gclasses)

        model_path = str(tmp_path / "test_model.pt")
        torch.save(
            {
                "model_name": "GCN",
                "lr": 0.01,
                "init_conv": init_conv,
                "rf_model": rf_model,
                "head0": head0,
                "head": head,
                "losses": [0.5, 0.4, 0.3],
                "valid_accs": [0.8, 0.85, 0.9],
                "test_accs": [0.78, 0.82, 0.88],
            },
            model_path,
        )

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        assert checkpoint["model_name"] == "GCN"
        assert checkpoint["lr"] == 0.01
        assert len(checkpoint["losses"]) == 3
        assert isinstance(checkpoint["init_conv"], GCN)
