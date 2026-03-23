import datetime
import json
import os
import timeit

from graph_classification.ais_timeseries_dataset import AISTimeseriesDataset

os.environ["DGLBACKEND"] = "pytorch"
import torch
import torch.nn.functional as F
from graph_classification.models import GCN
from graph_classification.heads import GraphClassificationHead
from graph_classification.utils import process_one, get_test_result, create_region_force_model, \
    get_elapsed_time_str, get_ais_datasets, create_ais_classification_model_df
from dgl.dataloading import GraphDataLoader
import numpy as np
import argparse


def train(device, train_ds: AISTimeseriesDataset, val_ds: AISTimeseriesDataset, test_ds: AISTimeseriesDataset,
          seed=0, model='GAT', lr=0.01, epochs=100, patience=0, batch_size=4000, model_path=None,
          pin_memory=True, num_workers=2):
    torch.manual_seed(seed)

    train_dataloader = GraphDataLoader(train_ds, sampler=None, batch_size=batch_size, drop_last=False,
                                       pin_memory=pin_memory, num_workers=num_workers)
    valid_dataloader = GraphDataLoader(val_ds, sampler=None, batch_size=batch_size, drop_last=False,
                                       pin_memory=pin_memory, num_workers=num_workers)
    test_dataloader = GraphDataLoader(test_ds, sampler=None, batch_size=batch_size, drop_last=False,
                                      pin_memory=pin_memory, num_workers=num_workers)
    dim_nfeats = train_ds.dim_nfeats
    gclasses = train_ds.gclasses
    print(f'training samples:{len(train_ds)}; validation samples:{len(val_ds)}; testing samples:{len(test_ds)}')
    ######################################################################
    # Initial convolutional layer
    init_conv = GCN(dim_nfeats, gclasses, depth=2).to(device)
    rf_model, hidden = create_region_force_model(device, dim_nfeats, model)
    head0 = GraphClassificationHead(gclasses, gclasses).to(device)
    head = GraphClassificationHead(hidden, gclasses).to(device)
    # noinspection PyUnboundLocalVariable
    optimizer = torch.optim.Adam(
        list(rf_model.parameters()) + list(head.parameters()) + list(init_conv.parameters()) + list(head0.parameters()),
        lr=lr)
    print(f'number of parameters in rf net: {sum(p.numel() for p in rf_model.parameters() if p.requires_grad)}')
    ######################################################################
    T = int(3)
    dt = 1. / T
    best_valid_acc = 0.
    best_test_acc = 0.
    best_epoch = 0.
    losses = []
    valid_accs = []
    test_accs = []
    data_key = 'attr'
    for epoch in range(epochs):
        start_time = timeit.default_timer()
        rf_model.train()
        init_conv.train()
        epoch_losses = []
        for batched_graph, labels in train_dataloader:
            batched_graph, labels = batched_graph.to(device), labels.to(device)
            pred = process_one(device, train_ds, batched_graph, head0, head, init_conv, data_key, rf_model, T, dt)
            loss = F.cross_entropy(pred, labels.type(torch.LongTensor if device == 'cpu' else torch.cuda.LongTensor))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(np.mean(epoch_losses))
        epoch_time = timeit.default_timer() - start_time
        # validation step
        rf_model.eval()
        init_conv.eval()
        with torch.no_grad():
            valid_acc = get_test_result(device, valid_dataloader, val_ds, head0, head, init_conv, data_key, rf_model,
                                        T, dt)
            valid_accs.append(valid_acc)

            test_acc = get_test_result(device, test_dataloader, test_ds, head0, head, init_conv, data_key, rf_model,
                                       T, dt)
            test_accs.append(test_acc)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_epoch = epoch
                if model_path:
                    print(
                        f'**epoch:{epoch}, time: {get_elapsed_time_str(epoch_time)}, loss:{np.mean(losses):.5f}, best val acc:{best_valid_acc:.5f}, best test acc:{best_test_acc:.5f}')
                    torch.save(
                        {'model_name': model, 'lr': lr, 'init_conv': init_conv,
                         'rf_model': rf_model, 'head0': head0,
                         'head': head, 'losses': losses,
                         'valid_accs': valid_accs, 'test_accs': test_accs},
                        model_path)
            else:
                if 0 < patience <= epoch - best_epoch:
                    print(
                        f'Epoch {epoch}: Validation score not improved in {patience} epocs (since epoch {best_epoch}). Terminating training.')
                    break
        if epoch % 10 == 0:
            print(
                f'  epoch:{epoch}, time: {get_elapsed_time_str(epoch_time)}, loss:{np.mean(losses):.5f}, best val acc:{best_valid_acc:.5f}, best test acc:{best_test_acc:.5f}')
    return best_test_acc, losses, valid_accs, test_accs


def main():
    parser = argparse.ArgumentParser(description='graph classification task')
    parser.add_argument('--data_folder', type=str, default='../data/',
                        help='folder for AIS data')
    parser.add_argument('--model_path', type=str, default='../results',
                        help='folder for AIS data')
    parser.add_argument('--models', type=str, default='GCN, GSG, GAT',
                        help='comma separated list of graph network to iterate over (GCN, GSG, GAT)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--lrs', type=str, default='5e-2, 3e-2, 1e-2',
                        help='comma separated learning rates to iterate over')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--hidden', type=int, default=32, help='size of hiden layers')
    parser.add_argument('--batch_size', type=int, default=600, help='batch size')
    parser.add_argument('--patience', type=int, default=200,
                        help='terminate if validation has not improved for this many epochs')
    parser.add_argument('--bootstrap_index', type=int, default=None,
                        help='part of total dataset to use for training, validation, and testing (0-49). If None, use all data')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin_memory for dataloader')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers for dataloader')
    args = parser.parse_args()

    # Check that the data folder exists
    if not os.path.exists(args.data_folder):
        raise FileNotFoundError(f'data_folder does not exist: {args.data_folder}')

    learning_rates = [float(lr) for lr in args.lrs.strip('[]').split(',')]
    models = [m.strip() for m in args.models.strip('[]').split(',')]
    device_ = 'cpu' if not torch.cuda.is_available() else 'cuda:' + str(args.gpu)
    print(f'\ndevice: {device_}')
    print(
        f'\nmodels: {models}, learning rates: {learning_rates}, epochs: {args.epochs}, hidden: {args.hidden}, batch size: {args.batch_size}, patience: {args.patience}')

    if args.model_path and not os.path.exists(args.model_path):
        print(f'Creating folder: {args.model_path}')
        os.mkdir(args.model_path)
    jobid = os.environ.get("SLURM_JOB_ID", '')
    pid = os.getpid()
    model_res = dict()
    bootstrap_idx = args.bootstrap_index
    train_ds, val_ds, test_ds = get_ais_datasets(args.data_folder, bootstrap_idx)
    print(f'\ntrain dataset: {train_ds}')
    for model in models:
        for lr in learning_rates:
            print(f'\n----------------------')
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            model_path = os.path.join(args.model_path,
                                      f'ais_classification_{timestamp}_bidx_{bootstrap_idx}_{model}_lr_{lr}_pid_{pid}_jobid_{jobid}.pt')
            print(f'model: {model}, learning rate: {lr}, model path: {model_path}')
            best_test_acc, losses, valid_accs, test_accs = train(device=device_, seed=0, train_ds=train_ds,
                                                                 val_ds=val_ds, test_ds=test_ds,
                                                                 model=model, lr=lr,
                                                                 epochs=args.epochs, patience=args.patience,
                                                                 batch_size=args.batch_size,
                                                                 model_path=model_path,
                                                                 pin_memory=args.pin_memory,
                                                                 num_workers=args.num_workers)
            model_res[f'{model}, _lr_{lr}_bootstrap_idx_{bootstrap_idx}'] = {'model': model,
                                                                             'bootstrap_idx': bootstrap_idx,
                                                                             'learning_rate': lr, 'acc': best_test_acc}
            print(f'model:{model}, learning rate: {lr}, best test accuracy: {best_test_acc:.5f}')

    print(f'\n----------------------')
    print(f'Results:')
    for key, value in model_res.items():
        print(
            f'{key}: best test accuracy: {value["acc"]:.5f}')
    # print result df
    df = create_ais_classification_model_df(model_res, model_key='model', lr_key='learning_rate', acc_key='best_test_acc')
    print(df)
    # saving model results
    model_res_path = os.path.join(args.model_path, f'ais_classification_model_results.json')
    with open(model_res_path, 'w') as f:
        json.dump(model_res, f)


if __name__ == '__main__':
    main()
    exit(0)
