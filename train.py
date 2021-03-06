from model import *
from dataset import *
import pickle as pk
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Regional
from dataset import RegionalDataset

import numpy as np
import copy

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

def random_split_target_dataset(obs_fn: str = './data/obs_data_target.pkl', n_test_station = 30):
    with open(obs_fn, 'rb') as f:
        obs = pk.load(f)
    test_indices = np.random.choice(len(obs), n_test_station, replace = False)
    test_mask = np.zeros(len(obs), dtype = bool)
    test_mask[test_indices] = True 
    train_stations, test_stations = list(np.array(list(obs.keys()))[~test_mask]), list(np.array(list(obs.keys()))[test_mask])
    return train_stations, test_stations

def temporal_split(length, n_test_date = 365):
    test_mask = np.zeros(length, dtype = bool)
    test_mask[-n_test_date:] = True
    return np.arange(length)[~test_mask], np.arange(length)[test_mask]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = 'Model name.')
    parser.add_argument('-e', '--num_epoch', type = int, help = 'The number of epoches for which the model will be trained.', default = 100)
    parser.add_argument('-bs', '--batch_size', type = int, help = 'Batch size for training.', default = 64)
    parser.add_argument('-lr', '--lr_config', nargs = 3, help = 'Learning rate configuration: [learning_rate, gamma, step_size].', default = [2e-3, 0.9, 256])
    parser.add_argument('--validate_every', type = int, help = 'Validate every x iterations.', default = 3)
    parser.add_argument('--save_threshold', type = float, help = 'Threshold of saving a best model.', default = 200.0)
    args = parser.parse_args()

    if not os.path.isdir(f'./{args.model}_models'):
        os.makedirs(f'./{args.model}_models')
    
    lr_config = _get_lr_config(*args.lr_config)

    train_stations, test_stations = random_split_target_dataset()
    train, test = RegionalDataset(train_stations), RegionalDataset(test_stations)
    assert len(train) == len(test)
    test.target_wrf_cmaq.normalizer = train.target_wrf_cmaq.normalizer
    
    train_dates, test_dates = temporal_split(len(train))
    train_subset, test_subset = torch.utils.data.Subset(train, train_dates), torch.utils.data.Subset(test, test_dates)
    model = Regional()

    print(f'Using {len(train_stations)} stations for training, {len(test_stations)} stations for validation.')
    print(f'Using {len(train_subset)} days for training, {len(test_subset)} days for validation.')
    
    print(f'Number of parameters in the model: {int(sum([np.prod(p.shape) for p in model.parameters()]))}')

    optimizer = torch.optim.Adam(model.parameters(), lr = lr_config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = lr_config['step_size'], gamma=lr_config['gamma'])

    best_val_loss = np.inf

    with open(f'./{args.model}_models/train.txt', 'w') as f:
        for st in train_stations:
            print(st, file = f)
    with open(f'./{args.model}_models/test.txt', 'w') as f:
        for st in test_stations:
            print(st, file = f)
    with open(f'./{args.model}_models/source_normalizers.pkl', 'wb') as f:
        pk.dump({st: ds.normalizer for st, ds in train.source.station_datasets.items()}, f)
    with open(f'./{args.model}_models/wrf_cmaq_normalizers.pkl', 'wb') as f:
        pk.dump(train.target_wrf_cmaq.normalizer, f)
    with open(f'./{args.model}_models/temporal_split.pkl', 'wb') as f:
        pk.dump({'train': train_dates, 'test': test_dates}, f)
    
    for i in range(args.num_epoch):
        epoch_loss = []
        dataloader, dist = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True), train.dist
        for j, (X0, X1, y) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = model(X0, X1, dist)
            y = torch.stack(list(y.values()), axis = 0)
            loss, count = batch_loss(y = y, pred = pred)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss.append((loss.item(), count.item()))
            print(f'Batch {j}, loss: {loss.item()/ count.item():.2f}')
        
        if i % args.validate_every == 0:
            model.eval()
            vdataloader, vdist = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size, shuffle=True), test.dist
            epoch_vloss = []
            with torch.no_grad():
                for j, (vX0, vX1, vy) in enumerate(vdataloader):
                    vpred = model(vX0, vX1, vdist)
                    vy = torch.stack(list(vy.values()), axis = 0)
                    vloss, vcount = batch_loss(y = vy, pred = vpred)
                    epoch_vloss.append((vloss.item(), vcount.item()))
            epoch_loss, epoch_vloss = _weighted_mean(epoch_loss), _weighted_mean(epoch_vloss)
            print(f'Iteration {i}, training loss: {epoch_loss:.2f}, validation loss: {epoch_vloss:.2f}.')
            model.train()

            if epoch_vloss <= min(best_val_loss, args.save_threshold):
                print(f'Saving model at iteration {i}...')
                best_val_loss = epoch_vloss
                torch.save(model, f'./{args.model}_models/model.nctmo')
        else:
            epoch_loss = _weighted_mean(epoch_loss)
            print(f'Iteration {i}, training loss: {epoch_loss:.2f}.')
    