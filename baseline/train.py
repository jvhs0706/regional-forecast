from model import *
from dataset import *
import pickle as pk
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import BaselineDataset
from model import Baseline

def _get_lr_config(*args):
    assert len(args) == 3
    return {
        'learning_rate': float(args[0]),
        'gamma': float(args[1]),
        'step_size': int(args[2])
    }

def batch_loss(y, pred):
    count = torch.sum(torch.logical_not(torch.isnan(y)))
    loss = torch.sum(torch.nan_to_num(y - pred) ** 2)
    return loss, count

def _weighted_mean(l):
    return sum(_[0] for _ in l)/sum(_[1] for _ in l)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = 'Model name.')
    parser.add_argument('-e', '--num_epoch', type = int, help = 'The number of epoches for which the model will be trained.', default = 256)
    parser.add_argument('-bs', '--batch_size', type = int, help = 'Batch size for training.', default = 64)
    parser.add_argument('-lr', '--lr_config', nargs = 3, help = 'Learning rate configuration: [learning_rate, gamma, step_size].', default = [1e-3, 0.9, 1024])
    parser.add_argument('--validate_every', type = int, help = 'Validate every x iterations.', default = 8)
    args = parser.parse_args()

    if not os.path.isdir(f'../{args.model}_models/baseline'):
        os.makedirs(f'../{args.model}_models/baseline')

    with open(f'../{args.model}_models/temporal_split.pkl', 'rb') as f:
        temporal_split = pk.load(f)
        train_dates, test_dates = temporal_split['train'], temporal_split['test']

    lr_config = _get_lr_config(*args.lr_config)

    with open('../data/obs_data_source.pkl', 'rb') as f:
        source_stations = list(pk.load(f).keys())

    for st in source_stations:
        fn = f'../{args.model}_models/baseline/{st}_model.nctmo'
        if os.path.isfile(fn):
            continue
        else:
            dataset = BaselineDataset(st)
            train_subset, test_subset = torch.utils.data.Subset(dataset, train_dates), torch.utils.data.Subset(dataset, test_dates)
            model = Baseline(len(dataset.obs_features), len(dataset.wrf_cmaq_features))
            print(f'Station {st}, number of params: {sum([np.prod(p.shape) for p in model.parameters()])}.')

            optimizer = torch.optim.Adam(model.parameters(), lr = lr_config['learning_rate'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = lr_config['step_size'], gamma=lr_config['gamma'])
            for i in range(args.num_epoch):
                epoch_loss = []
                dataloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
                for X0, X1, y in dataloader:
                    optimizer.zero_grad()
                    pred = model(X0, X1)
                    loss, count = batch_loss(y = y, pred = pred)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    epoch_loss.append((loss.item(), count.item()))
                
                if i % args.validate_every == 0:
                    model.eval()
                    vdataloader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size, shuffle=True)
                    epoch_vloss = []
                    with torch.no_grad():
                        for vX0, vX1, vy in vdataloader:
                            vpred = model(vX0, vX1)
                            vloss, vcount = batch_loss(y = vy, pred = vpred)
                            epoch_vloss.append((vloss.item(), vcount.item()))
                    print(f'Iteration {i}, training loss: {_weighted_mean(epoch_loss):.2f}, validation loss: {_weighted_mean(epoch_vloss):.2f}.')
                    model.train()
                else:
                    print(f'Iteration {i}, training loss: {_weighted_mean(epoch_loss):.2f}.')
            torch.save(model, fn)