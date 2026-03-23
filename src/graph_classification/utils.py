import os
import timeit
from enum import Enum
from typing import Tuple

import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from graph_classification.ais_timeseries_dataset import AISTimeseriesDataset
from graph_classification.models import GCN, GAT, GraphSAGE


class DS_Type(str, Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


def GraphLaplacian(adj, symmetric=True):
    # Compute the row sum of the adjacency matrix
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()  # Convert to dense to perform element-wise operations

    if symmetric:
        # Calculate the inverse square root of the degree matrix
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # Set infinities to 0

        # Create a sparse diagonal matrix for d_inv_sqrt
        indices = torch.arange(len(d_inv_sqrt), device=adj.device)
        d_mat_inv_sqrt = torch.sparse_coo_tensor(
            indices=torch.stack([indices, indices]),
            values=d_inv_sqrt,
            size=(len(d_inv_sqrt), len(d_inv_sqrt)),
            device=adj.device
        )

        # Compute the symmetric Laplacian L = D^(-1/2) * A * D^(-1/2)
        res = torch.sparse.mm(d_mat_inv_sqrt, adj)
        res = torch.sparse.mm(res, d_mat_inv_sqrt)

    else:
        # Calculate the inverse of the degree matrix
        d_inv = torch.pow(row_sum, -1)
        d_inv[torch.isinf(d_inv)] = 0.0  # Set infinities to 0

        # Create a sparse diagonal matrix for d_inv
        indices = torch.arange(len(d_inv), device=adj.device)
        d_mat_inv = torch.sparse_coo_tensor(
            indices=torch.stack([indices, indices]),
            values=d_inv,
            size=(len(d_inv), len(d_inv)),
            device=adj.device
        )

        # Compute the Laplacian L = D^(-1) * A
        res = torch.sparse.mm(d_mat_inv, adj)

    return res.coalesce()  # Ensure the sparse matrix is in canonical form


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_Laplacian(batched_graph, name, device):
    if name == 'MNIST' or name == 'CIFAR10':
        return get_Laplacian_withefeat(batched_graph, device)
    else:
        return get_Laplacian_noefeat(batched_graph, device)


def get_Laplacian_noefeat(batched_graph, device):
    us, vs = batched_graph.edges()

    # Create a dense tensor of ones on the appropriate device
    us = us.clone().detach().float().to(device)
    vs = vs.clone().detach().float().to(device)
    vals = torch.ones_like(us)

    # Create the adjacency matrix A as a sparse PyTorch tensor
    A = torch.sparse_coo_tensor(
        indices=torch.stack([us, vs]),
        values=vals,
        size=(batched_graph.num_nodes(), batched_graph.num_nodes())
    ).coalesce()  # coalesce to make sure the sparse tensor is in canonical form

    L = GraphLaplacian(A, False)

    return L.to(device)  # Ensure L is moved to the correct device (e.g., GPU)


def get_Laplacian_withefeat(batched_graph, device):
    us, vs = batched_graph.edges()  # Get edge indices
    vals = batched_graph.edata['feat'].view(-1).to(device)  # Edge features (ensure they are on the correct device)

    # Create the adjacency matrix A as a sparse PyTorch tensor
    A = torch.sparse_coo_tensor(
        indices=torch.stack([torch.tensor(us, device=device), torch.tensor(vs, device=device)]),
        values=vals,
        size=(batched_graph.num_nodes(), batched_graph.num_nodes())
    ).coalesce()  # Coalesce to ensure the sparse tensor is in canonical form

    # Compute the Graph Laplacian
    L = GraphLaplacian(A, symmetric=False)  # Pass A to the GraphLaplacian function (you may toggle symmetric as needed)

    return L.to(device)  # Return the Laplacian matrix on the correct device (GPU or CPU)


def transform_graph(batched_graph):
    batched_graph = dgl.remove_self_loop(batched_graph)
    batched_graph = dgl.add_reverse_edges(batched_graph)
    return batched_graph


def create_region_force_model(device, dim_nfeats, model) -> (torch.nn.Module, int):
    # Create the region force model
    if model == 'GCN':
        hidden = 64
        return GCN(dim_nfeats, hidden, depth=3).to(device), hidden
    if model == 'GSG':
        hidden = 64
        return GraphSAGE(dim_nfeats, hidden, depth=3).to(device), hidden
    if model == 'GAT':
        hidden = 32
        return GAT(dim_nfeats, hidden, depth=3).to(device), hidden
    raise ValueError(f'Unknown model: {model}')


def process_one(device, dataset, batched_graph, head0, head, init_conv, data_key, rf_model, T, dt):
    batched_graph.Laplacian = get_Laplacian(batched_graph, dataset, device)
    u = head0(batched_graph, init_conv(batched_graph, batched_graph.ndata[data_key].float()))
    rf = head(batched_graph, rf_model(batched_graph, batched_graph.ndata[data_key].float()))
    for t in range(T):
        u = torch.tanh(u + rf * dt)
    return u


def get_test_result(device, dataloader, dataset, head0, head, init_conv, data_key, rf_model, T, dt):
    num_correct = 0
    num_tests = 0
    for batched_graph, labels in dataloader:
        batched_graph, labels = batched_graph.to(device), labels.to(device)
        pred = process_one(device, dataset, batched_graph, head0, head, init_conv, data_key, rf_model, T, dt)
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)
    valid_acc = num_correct / num_tests
    return valid_acc


def get_ais_datasets(data_path: str, k=0, name: str = 'ais_graph_classification_dataset') -> Tuple[
    AISTimeseriesDataset, AISTimeseriesDataset, AISTimeseriesDataset]:
    ds_name = f'{name}_K_{k}'
    start = timeit.default_timer()
    train_file_name_x, train_file_name_y, val_file_name_x, val_file_name_y, test_file_name_x, test_file_name_y = None, None, None, None, None, None
    try:
        train_file_name_x, train_file_name_y, val_file_name_x, val_file_name_y, test_file_name_x, test_file_name_y = get_numpy_ds_files(
            data_path, k)
    except FileNotFoundError as e:
        print(f'Warning: {e}, will try loading from graph dataset instead')

    train_ds = AISTimeseriesDataset(name=f'{ds_name}_train', raw_x_file=train_file_name_x, raw_y_file=train_file_name_y,
                                    save_dir=data_path)
    val_ds = AISTimeseriesDataset(name=f'{ds_name}_val', raw_x_file=val_file_name_x, raw_y_file=val_file_name_y,
                                  save_dir=data_path)
    test_ds = AISTimeseriesDataset(name=f'{ds_name}_test', raw_x_file=test_file_name_x, raw_y_file=test_file_name_y,
                                   save_dir=data_path)
    print(f'create dgl dataset time: {get_elapsed_time_str(timeit.default_timer() - start)}')
    return train_ds, val_ds, test_ds


def get_ais_dataset(data_path: str, k: int, type: DS_Type,
                    name: str = 'ais_graph_classification_dataset') -> AISTimeseriesDataset:
    ds_name = f'{name}_K_{k}_{type.name}'
    train_file_name_x, train_file_name_y, val_file_name_x, val_file_name_y, test_file_name_x, test_file_name_y = get_numpy_ds_files(
        data_path, k)
    if type == DS_Type.TRAIN:
        return AISTimeseriesDataset(name=f'{ds_name}', raw_x_file=train_file_name_x, raw_y_file=train_file_name_y,
                                    save_dir=data_path)
    if type == DS_Type.VALIDATION:
        return AISTimeseriesDataset(name=f'{ds_name}', raw_x_file=val_file_name_x, raw_y_file=val_file_name_y,
                                    save_dir=data_path)
    if type == DS_Type.TEST:
        return AISTimeseriesDataset(name=f'{ds_name}', raw_x_file=test_file_name_x, raw_y_file=test_file_name_y,
                                    save_dir=data_path)


def get_numpy_ds_files(data_path: str, k) -> Tuple[str, str, str, str, str, str]:
    train_file_name = os.path.join(data_path, f'ais_graph_classification_dataset_K_{k}_train_X.npy')
    train_file_name_y = os.path.join(data_path, f'ais_graph_classification_dataset_K_{k}_train_y.npy')
    val_file_name = os.path.join(data_path, f'ais_graph_classification_dataset_K_{k}_val_X.npy')
    val_file_name_y = os.path.join(data_path, f'ais_graph_classification_dataset_K_{k}_val_y.npy')
    test_file_name = os.path.join(data_path, f'ais_graph_classification_dataset_K_{k}_test_X.npy')
    test_file_name_y = os.path.join(data_path, f'ais_graph_classification_dataset_K_{k}_test_y.npy')
    if not (os.path.exists(train_file_name) and os.path.exists(val_file_name) and os.path.exists(test_file_name)
            and os.path.exists(train_file_name_y) and os.path.exists(val_file_name_y) and os.path.exists(
                test_file_name_y)):
        ais_X, ais_y, split_idx = get_ais_ts_data(data_path, 'X_ts12.npy', 'y_ts12.npy')
        x_train, y_train, x_val, y_val, x_test, y_test = ais_data_split(k, split_idx, ais_X, ais_y)
        save_numpy_data(x_train, train_file_name, y_train, train_file_name_y)
        save_numpy_data(x_val, val_file_name, y_val, val_file_name_y)
        save_numpy_data(x_test, test_file_name, y_test, test_file_name_y)
    return train_file_name, train_file_name_y, val_file_name, val_file_name_y, test_file_name, test_file_name_y


def get_elapsed_time_str(elapsed_time: float) -> str:
    minutes, seconds = divmod(elapsed_time, 60)
    if minutes == 0:
        return f'{seconds:.2f} seconds'
    return f'{int(minutes)} minutes and {seconds:.2f} seconds'


def get_ais_ts_data(folder, infile, labelfile):
    # Load input vdc data
    file = os.path.join(folder, infile)
    if not os.path.exists(file):
        raise FileNotFoundError(f'File not found: {file}')
    x_data = np.load(file).astype(np.float32)

    # Load label data
    file = os.path.join(folder, labelfile)
    y_data = np.abs(np.load(file)).astype(np.float32)

    # Load the idx data
    file = os.path.join(folder, labelfile.replace('y_', 'bidx_'))
    bidx = np.load(file)

    assert np.all((y_data >= 0) & (y_data <= 1))
    print(f'{x_data.shape=}, {y_data.shape=}, {bidx.shape=}')

    return x_data, y_data, bidx


def ais_data_split(k: int, bidx: np.ndarray, x_data: np.ndarray, y_data: np.ndarray):
    # Split the data into training, validation and test sets
    if k is None:
        train_indices = set(np.nonzero(bidx == 1)[1])
        val_indices = set(np.nonzero(bidx == 2)[1])
        test_indices = set(np.nonzero(bidx == 3)[1])
    else:
        train_indices = set(np.nonzero(bidx[k] == 1)[0])
        val_indices = set(np.nonzero(bidx[k] == 2)[0])
        test_indices = set(np.nonzero(bidx[k] == 3)[0])

    # Remove overlapping indices
    val_indices = val_indices - train_indices
    test_indices = test_indices - train_indices - val_indices

    # Convert sets to sorted lists for consistent indexing
    i_train = sorted(train_indices)
    i_val = sorted(val_indices)
    i_test = sorted(test_indices)

    x_train = x_data[i_train]
    y_train = y_data[i_train]

    x_val = x_data[i_val]
    y_val = y_data[i_val]

    x_test = x_data[i_test]
    y_test = y_data[i_test]

    return x_train, y_train, x_val, y_val, x_test, y_test


def save_numpy_data(x: np.ndarray, x_path: str, y: np.ndarray, y_path: str):
    np.save(x_path, x)
    np.save(y_path, y)
    return


def create_ais_classification_model_df(results, model_key='model', lr_key='learning_rate',
                                       acc_key: str = 'best_test_acc', bootstrap_key='bootstrap_idx') -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(results, orient='index')
    if bootstrap_key in list(results.values())[0].keys():
        pivot_df = df.pivot_table(index=[model_key, bootstrap_key], columns=lr_key, values=acc_key)
    else:
        pivot_df = df.pivot_table(index=model_key, columns=lr_key, values=acc_key)
    pivot_df.columns = [f'Test acc for lr {col}' for col in pivot_df.columns]
    return pivot_df


def plot_metrics(models, losses, valid_accs, test_accs):
    num_rows = len(models)
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    # If num_rows is 1, axes is a 1D array. Convert to 2D for consistency.
    if num_rows == 1:
        axes = axes[np.newaxis, :]  # This ensures axes has 2D shape

    for row in range(num_rows):
        # Plot losses
        axes[row, 0].plot(losses[row], marker='o', linestyle='-', color='b')
        axes[row, 0].set_title(f'Losses for model {models[row]}')
        axes[row, 0].set_xlabel('Epoch')
        axes[row, 0].set_ylabel('Loss')
        axes[row, 0].grid(True)

        # Plot validation accuracies
        axes[row, 1].plot(valid_accs[row], marker='o', linestyle='-', color='g')
        axes[row, 1].set_title(f'Validation Accuracies for model {models[row]}')
        axes[row, 1].set_xlabel('Epoch')
        axes[row, 1].set_ylabel('Validation Accuracy')
        axes[row, 1].grid(True)

        # Plot test accuracies
        axes[row, 2].plot(test_accs[row], marker='o', linestyle='-', color='r')
        axes[row, 2].set_title(f'Test Accuracies for model {models[row]}')
        axes[row, 2].set_xlabel('Epoch')
        axes[row, 2].set_ylabel('Test Accuracy')
        axes[row, 2].grid(True)

    plt.tight_layout()
    plt.show()
