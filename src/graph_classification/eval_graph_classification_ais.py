import glob
import os
from typing import List, Tuple

from graph_classification.ais_timeseries_dataset import AISTimeseriesDataset

os.environ["DGLBACKEND"] = "pytorch"
import torch
from graph_classification.utils import get_test_result, DS_Type, get_ais_dataset, create_ais_classification_model_df
from dgl.dataloading import GraphDataLoader
import argparse


def test(device, test_ds: AISTimeseriesDataset, model_path, batch_size=4000) -> Tuple[
    str, float, float, List[float], List[float], List[float]]:
    test_dataloader = GraphDataLoader(test_ds, sampler=None, batch_size=batch_size, drop_last=False)
    checkpoint = torch.load(model_path, map_location=device)
    model_name, lr = checkpoint['model_name'], checkpoint['lr']
    init_conv, rf_model, head0, head = checkpoint['init_conv'], checkpoint['rf_model'], checkpoint['head0'], checkpoint[
        'head']
    losses, valid_accs, test_accs = checkpoint['losses'], checkpoint['valid_accs'], checkpoint['test_accs']
    # print(f'losses:{losses}, valid_accs:{valid_accs}, test_accs:{test_accs}')
    T = int(3)
    dt = 1. / T
    data_key = 'attr'
    with torch.no_grad():
        test_acc = get_test_result(device, test_dataloader, test_ds, head0.to(device), head.to(device),
                                   init_conv.to(device), data_key, rf_model.to(device), T, dt)
    return model_name, lr, test_acc, losses, valid_accs, test_accs


def main():
    parser = argparse.ArgumentParser(description='graph classification task')
    parser.add_argument('--data_folder', type=str, default='../data', help='Path to datasets')
    parser.add_argument('--k', type=int, default=None,
                        help='bidx part of total dataset to use for training, validation, and testing (0-49)')
    parser.add_argument('--batch_size', type=int, default=4000, help='batch size')
    parser.add_argument('--model_path', type=str, default='../results',
                        help='folder for AIS graph classification models')
    args = parser.parse_args()

    stored_model_path = args.model_path
    if stored_model_path and not os.path.exists(stored_model_path):
        raise FileNotFoundError(f'model_path does not exist: {args.model_path}')

    models = glob.glob(os.path.join(stored_model_path, '*.pt'))
    if len(models) == 0:
        raise FileNotFoundError(f'No models found in {stored_model_path}')

    print(f'\nLoading from folder: {args.data_folder}, k={args.k}')
    test_ds = get_ais_dataset(args.data_folder, args.k, DS_Type.TEST, 'ais_timeseries_test')
    print(f'\ntest dataset: {test_ds}')

    results = dict()
    for model_file in models:
        print(f'----------------------')
        print(f'\nTesting model: {model_file}')
        model_type, learning_rate, test_acc, losses, val_accs, test_accs = test('cpu', test_ds, model_file,
                                                                                batch_size=args.batch_size)
        print(f'Test results: {test_acc}')
        results[model_file] = {'model': model_type, 'learning_rate': learning_rate, 'test_acc': test_acc,
                               'losses': losses, 'val_accs': val_accs, 'test_accs': test_accs}

    print(f'----------------------')
    print(f'\nBest Model:')
    best_model_file, best_model_data = max(results.items(), key=lambda item: item[1]['test_acc'])
    print(
        f"{best_model_data['model']} (learning rate: {best_model_data['learning_rate']}, results: {best_model_data['test_acc']})\n")

    result_df = create_ais_classification_model_df(results)
    print(result_df)


if __name__ == '__main__':
    main()
    exit(0)
