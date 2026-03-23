"""Tests for verifying all imports and module availability."""
import pytest


def _dgl_available():
    try:
        import dgl  # noqa: F401
        return True
    except (ImportError, FileNotFoundError, Exception):
        return False


dgl_skip = pytest.mark.skipif(not _dgl_available(), reason="dgl not available in this environment")


class TestImports:
    """Verify all project modules are importable."""

    @dgl_skip
    def test_import_dgl(self):
        import dgl
        assert dgl.__version__

    def test_import_torch(self):
        import torch
        assert torch.__version__

    @dgl_skip
    def test_import_models(self):
        from graph_classification.models import GCN, GAT, GraphSAGE  # noqa: F401

    @dgl_skip
    def test_import_heads(self):
        from graph_classification.heads import (  # noqa: F401
            GraphClassificationHead,
            NodeClassificationHead,
        )

    @dgl_skip
    def test_import_dataset(self):
        from graph_classification.ais_timeseries_dataset import AISTimeseriesDataset  # noqa: F401

    @dgl_skip
    def test_import_utils(self):
        from graph_classification.utils import (  # noqa: F401
            GraphLaplacian,
            create_region_force_model,
            get_elapsed_time_str,
            get_test_result,
            plot_metrics,
            process_one,
            transform_graph,
        )

    @dgl_skip
    def test_import_train(self):
        from graph_classification.train_graph_classification_ais import train  # noqa: F401

    @dgl_skip
    def test_import_eval(self):
        from graph_classification.eval_graph_classification_ais import test  # noqa: F401
