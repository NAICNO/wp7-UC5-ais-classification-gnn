import os.path

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, save_info, load_graphs, load_info


class AISTimeseriesDataset(DGLDataset):
    def __init__(self, name: str, raw_x_file: str, raw_y_file: str, save_dir=None, transform=None):
        self.graphs = []
        self.labels = []
        self.dim_nfeats = 0
        self.gclasses = 0
        self._raw_x_file = raw_x_file
        self._raw_y_file = raw_y_file
        raw_dir = os.path.dirname(raw_x_file) if raw_x_file is not None else None
        super(AISTimeseriesDataset, self).__init__(name=name, raw_dir=raw_dir, save_dir=save_dir,
                                                   force_reload=False, transform=transform)

    def process(self):
        if self._raw_x_file is None or self._raw_y_file is None:
            raise ValueError('Raw data files are not set.')
        # Load the dataset from the raw data
        node_features_array, labels = get_ais_ts_data(self._raw_x_file, self._raw_y_file)
        self.create_from_numpy(node_features_array, labels)

    def create_from_numpy(self, node_features_array, labels):
        # node features  Shape: (N, 3, num_timesteps)
        # labels  Shape: (N, )
        self.graphs = []
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.dim_nfeats = node_features_array.shape[1]
        self.gclasses = len(np.unique(labels))
        # Create a sequential edge list: (0->1, 1->2, ..., 10->11)
        num_timesteps = node_features_array.shape[2]
        edge_list = [(i, i + 1) for i in range(num_timesteps - 1)]
        # Iterate over the samples (N)
        for i in range(node_features_array.shape[0]):
            # Get the features for sample i and transpose it to (12, 3)
            node_features = torch.tensor(node_features_array[i].T, dtype=torch.float32)  # Shape: (12, 3)
            # Create a graph with the edge list
            graph = dgl.graph(edge_list)
            # Set the node features
            graph.ndata['attr'] = node_features
            # add self loop to avoid '0-in-degree nodes in the graph' error
            graph = dgl.add_self_loop(graph)
            self.graphs.append(graph)

    def has_cache(self):
        return os.path.exists(self.save_path)

    def save(self):
        if not os.path.exists(self.save_path) and self.save_dir:
            os.makedirs(self.save_path)
        # dgl save_graphs requires labels: dict[str, Tensor]
        label_dict = {"labels": self.labels}
        save_graphs(self.graph_path, self.graphs, label_dict)
        info_dict = {
            "gclasses": self.gclasses,
            "dim_nfeats": self.dim_nfeats,
        }
        save_info(str(self.info_path), info_dict)

    @property
    def graph_path(self):
        return os.path.join(
            self.save_path, f'ais_ts_{self.name}_{self.hash}.bin'
        )

    @property
    def info_path(self):
        return os.path.join(
            self.save_path, f'ais_ts_info_{self.name}_{self.hash}.pkl'
        )

    def load(self):
        # Load the graphs, labels, and metadata from the file
        graphs, label_dict = load_graphs(str(self.graph_path))
        info_dict = load_info(str(self.info_path))

        self.graphs = graphs
        self.labels = label_dict['labels']
        self.gclasses = info_dict['gclasses']
        self.dim_nfeats = info_dict['dim_nfeats']

    def _download(self):
        # No download required
        pass

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]

        if self._transform:
            graph = self._transform(graph)

        return graph, label

    def __repr__(self):
        return (f"{self.__class__.__name__}(\n"
                f"  Number of graphs: {len(self.graphs)}\n"
                f"  Node feature dimension: {self.dim_nfeats}\n"
                f"  Number of label classes: {self.gclasses}\n)")


def get_ais_ts_data(x_file: str, y_file) -> (np.ndarray, np.ndarray):
    # Load input vdc data
    x_data = np.load(x_file).astype(np.float32)
    # Load label data
    y_data = np.abs(np.load(y_file)).astype(np.float32)

    assert np.all((y_data >= 0) & (y_data <= 1))
    return x_data, y_data
