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

import matplotlib.pyplot as plt
from geopy.distance import geodesic
from plot_stations import get_border

import pandas as pd

def MAE(pred, y, axis = 0):
    return np.nanmean(np.abs(pred - y), axis = axis)

def RMSE(pred, y, axis = 0):
    return np.sqrt(np.nanmean((pred - y)**2, axis = axis))

def R(pred, y, axis = 0):
    mask = ~np.isnan(y)
    n = mask.sum(axis = axis, keepdims = True)
    pred_mean = np.sum(pred * mask, axis = axis, keepdims = True) / n
    y_mean = np.nanmean(y, axis = axis, keepdims = True)
    r = np.nansum((pred - pred_mean) * (y - y_mean), axis = axis)/ np.sqrt(np.sum(mask * (pred - pred_mean)**2, axis = axis) * np.nansum((y - y_mean)**2, axis = axis))
    return r

def regional_reliability(source_lat_lon, weight_decay, lat_range, lon_range):
    x, y = np.linspace(*lon_range, 100), np.linspace(*lat_range, 100)
    X, Y = np.meshgrid(x, y)
    out = np.zeros_like(X)
    for decay_factor, (lat, lon) in zip(weight_decay, source_lat_lon.values()):
        dist = np.apply_along_axis(lambda z: geodesic((z[1], z[0]), (lat, lon)).km, axis = 0, arr = np.stack([X, Y]))
        out += np.exp(-decay_factor * dist)
    return X, Y, out        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = 'Model name.')
    parser.add_argument('mode', choices= ['overall', 'reliability', 'metrics'])
    parser.add_argument('-bs', '--batch_size', type = int, help = 'Batch size for loading data.', default = 64)
    args = parser.parse_args()
    
    # model loading
    model = torch.load(f'./{args.model}_models/model.nctmo')
    model.eval()
    decay = model.gcn.distance_decay.detach().numpy()
    
    with open(f'./{args.model}_models/train.txt', 'r') as f:
        train_stations = f.read().splitlines()
    with open(f'./{args.model}_models/test.txt', 'r') as f:
        test_stations = f.read().splitlines()
    with open('./data/lat_lon_source.pkl', 'rb') as f:
        source_lat_lon = pk.load(f)
        source_stations = list(source_lat_lon.keys())   
    with open('./data/lat_lon_target.pkl', 'rb') as f:
        target_lat_lon = pk.load(f) 
    with open(f'./{args.model}_models/source_normalizers.pkl', 'rb') as f:
        source_normalizers = pk.load(f)
    with open(f'./{args.model}_models/wrf_cmaq_normalizers.pkl', 'rb') as f:
        wrf_cmaq_normalizers = pk.load(f)
    with open(f'./{args.model}_models/temporal_split.pkl', 'rb') as f:
        split = pk.load(f)
        train_dates, test_dates = split['train'], split['test']

    train, test = RegionalDataset(train_stations), RegionalDataset(test_stations)
    train_subset, test_subset = torch.utils.data.Subset(train, train_dates), torch.utils.data.Subset(test, test_dates)
    test.target_wrf_cmaq.normalizer = wrf_cmaq_normalizers
    for st in source_stations:
        train.source.station_datasets[st].normalizer = source_normalizers[st]
        test.source.station_datasets[st].normalizer = source_normalizers[st]

    train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=False)
    train_pred = {st: ([], [], None) for st in train_stations}
    train_target_features = [train.target_wrf_cmaq.features.index('FSPMC'), train.target_wrf_cmaq.features.index('O3')]
    with torch.no_grad():
        for j, (X0, X1, obs) in enumerate(train_dataloader):
            pred = model(X0, X1, train.dist).detach()
            # print(pred.shape, len(obs))
            for st, arr, y in zip(train_stations, pred, obs.values()):
                train_pred[st][0].append(arr)
                train_pred[st][1].append(y)
            print(f'Testing on training set, batch {j}...')
    
    for st in train_stations:
        ds = train.target_wrf_cmaq.station_datasets[st]
        train_pred[st] = (np.concatenate(train_pred[st][0]), np.concatenate(train_pred[st][1]), \
            np.stack(
                [
                    ds.wrf_cmaq[ds.history: ds.history + len(ds), train_target_features[0], :],
                    1000 * ds.wrf_cmaq[ds.history: ds.history + len(ds), train_target_features[1], :]
                ], axis = 1
            )[train_dates]
        )

    test_dataloader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
    test_pred = {st: ([], [], None) for st in test_stations}
    test_target_features = [test.target_wrf_cmaq.features.index('FSPMC'), test.target_wrf_cmaq.features.index('O3')]
    assert train_target_features == test_target_features
    with torch.no_grad():
        for j, (X0, X1, obs) in enumerate(test_dataloader):
            pred = model(X0, X1, test.dist).detach()
            # print(pred.shape, len(obs))
            for st, arr, y in zip(test_stations, pred, obs.values()):
                test_pred[st][0].append(arr)
                test_pred[st][1].append(y)
            print(f'Testing on test set, batch {j}...')

    for st in test_stations:
        ds = test.target_wrf_cmaq.station_datasets[st]
        test_pred[st] = (np.concatenate(test_pred[st][0]), np.concatenate(test_pred[st][1]), \
            np.stack(
                [
                    ds.wrf_cmaq[ds.history: ds.history + len(ds), test_target_features[0], :],
                    1000 * ds.wrf_cmaq[ds.history: ds.history + len(ds), test_target_features[1], :]
                ], axis = 1
            )[test_dates]
        )

    # if args.baseline:
    #     baseline_source_predictions = {}
    #     for st in source_stations:
    #         baseline_dataset = BaselineDataset(st)
    #         baseline_dataloader = torch.utils.data.DataLoader(baseline_dataset, batch_size=args.batch_size, shuffle=False)
    #         baseline_model = torch.load(f'./{args.model}_models/baseline/{st}_model.nctmo')
    #         baseline_model.eval()

    #         baseline_pred = ([], None)
    #         for X0, X1, _ in baseline_dataloader:
    #             baseline_pred[0].append(baseline_model(X0, X1).detach().numpy()) 

    #         wrf_cmaq_target_indices = [baseline_dataset.wrf_cmaq_features.index('FSPMC'), baseline_dataset.wrf_cmaq_features.index('O3')]
    #         baseline_pred = \
    #             (   
    #                 np.concatenate(baseline_pred[0], axis = 0), \
    #                 baseline_dataset.wrf_cmaq[baseline_dataset.history: baseline_dataset.history + len(baseline_dataset), wrf_cmaq_target_indices, :] * np.array([1, 1000]).reshape(1, 2, 1)
    #             )
    #         baseline_source_predictions[st] = baseline_pred

    #     for st in test_stations:
    #         if st not in source_stations:
    #             weights = {s: 1/geodesic(source_lat_lon[s], target_lat_lon[st]).km for s in source_stations}
    #             n = sum([weights[s] for s in source_stations])
    #             weights = {s: weights[s] / n for s in source_stations} 
    #             baseline_pred = test_pred[st][2].copy()
    #             for s in source_stations:
    #                 baseline_pred += weights[s] * (baseline_source_predictions[s][0][test_dates] - baseline_source_predictions[s][1][test_dates])
    #             test_pred[st] = (*test_pred[st], baseline_pred)

    if args.mode == 'overall':
        for st in train_stations:
            print(f'Station {st}:')
            print(f'Model MAE: {MAE(train_pred[st][0], train_pred[st][1], axis = (0, 2))}, CMAQ MAE: {MAE(train_pred[st][2], train_pred[st][1], axis = (0, 2))}')
            print(f'Model RMSE: {RMSE(train_pred[st][0], train_pred[st][1], axis = (0, 2))}, CMAQ RMSE: {RMSE(train_pred[st][2], train_pred[st][1], axis = (0, 2))}')
            print()

        for st in test_stations:
            print(f'Station {st}:')
            print(f'Model MAE: {MAE(test_pred[st][0], test_pred[st][1], axis = (0, 2))}, CMAQ MAE: {MAE(test_pred[st][2], test_pred[st][1], axis = (0, 2))}')
            print(f'Model RMSE: {RMSE(test_pred[st][0], test_pred[st][1], axis = (0, 2))}, CMAQ RMSE: {RMSE(test_pred[st][2], test_pred[st][1], axis = (0, 2))}')
            print()

        overall_train_MAE = MAE(np.concatenate([train_pred[st][0] for st in train_stations], axis = 0), np.concatenate([train_pred[st][1] for st in train_stations], axis = 0), axis = (0, 2))
        overall_train_RMSE = RMSE(np.concatenate([train_pred[st][0] for st in train_stations], axis = 0), np.concatenate([train_pred[st][1] for st in train_stations], axis = 0), axis = (0, 2))
        overall_test_MAE = MAE(np.concatenate([test_pred[st][0] for st in test_stations], axis = 0), np.concatenate([test_pred[st][1] for st in test_stations], axis = 0), axis = (0, 2))
        overall_test_RMSE = RMSE(np.concatenate([test_pred[st][0] for st in test_stations], axis = 0), np.concatenate([test_pred[st][1] for st in test_stations], axis = 0), axis = (0, 2))
        print(F'Train MAE: {overall_train_MAE}')
        print(F'Train RMSE: {overall_train_RMSE}')
        print(F'Test MAE: {overall_test_MAE}')
        print(F'Test RMSE: {overall_test_RMSE}')

    elif args.mode == 'reliability':
        plt.contourf(*regional_reliability(source_lat_lon, decay, (21, 24), (112, 115)))
        plt.colorbar()
        plt.title('Reliability')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.plot([loc[1] for loc in source_lat_lon.values()], [loc[0] for loc in source_lat_lon.values()], 'or', markersize = 3, label = 'Source stations')
        plt.plot([loc[1] for st, loc in target_lat_lon.items() if st in train_stations], [loc[0] for st, loc in target_lat_lon.items() if st in train_stations], 'og', markersize = 2, label = 'Target training stations')
        plt.plot([loc[1] for st, loc in target_lat_lon.items() if st in test_stations], [loc[0] for st, loc in target_lat_lon.items() if st in test_stations], 'ob', markersize = 2, label = 'Target held-out stations')
        borders = get_border(21, 24, 112, 115)
        for (blon, blat) in borders:
            plt.plot(blon, blat, 'k', linewidth=1)
        plt.legend()
        plt.savefig(f'./{args.model}_models/reliability_map.png')
        plt.close()
        
        train_mae, train_rmse = [], []
        for st, pred in train_pred.items():
            train_mae.append(MAE(train_pred[st][0], train_pred[st][1], axis = (0, 2)))
            train_rmse.append(RMSE(train_pred[st][0], train_pred[st][1], axis = (0, 2)))
            
        train_mae, train_rmse = np.stack(train_mae, axis = 0), np.stack(train_rmse, axis = 0)
        train_reliability = np.exp(-decay[:, None] * train.dist.detach().numpy()).sum(axis = 0)
        
        test_mae, test_rmse = [], []
        for st, pred in test_pred.items():
            test_mae.append(MAE(test_pred[st][0], test_pred[st][1], axis = (0, 2)))
            test_rmse.append(RMSE(test_pred[st][0], test_pred[st][1], axis = (0, 2)))
            
        test_mae, test_rmse = np.stack(test_mae, axis = 0), np.stack(test_rmse, axis = 0)
        test_reliability = np.exp(-decay[:, None] * test.dist.detach().numpy()).sum(axis = 0)

        fig, axes = plt.subplots(1, 2, figsize = (10, 4))

        for i, (sp, display_name, u) in enumerate(zip(['fspmc', 'o3'], ['$\mathrm{PM}_{2.5}$', '$\mathrm{O}_3$'], ['$\mu g/m^3$', 'ppbv'])): 
            axes[i].scatter(train_reliability, train_mae[:, i], c = '#ff0000', label = 'Train')
            axes[i].scatter(test_reliability, test_mae[:, i], c = '#0000ff', label = 'Test')
            axes[i].legend()
            axes[i].set_xlabel('Reliability')
            axes[i].set_ylabel(f'MAE ({u})')
            axes[i].set_title(display_name)
        
        fig.suptitle('Reliability vs MAE')
        plt.savefig(f'./{args.model}_models/mae_vs_reliability.png')
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize = (10, 4))

        for i, (sp, display_name, u) in enumerate(zip(['fspmc', 'o3'], ['$\mathrm{PM}_{2.5}$', '$\mathrm{O}_3$'], ['$\mu g/m^3$', 'ppbv'])): 
            axes[i].scatter(train_reliability, train_rmse[:, i], c = '#ff0000', label = 'Train')
            axes[i].scatter(test_reliability, test_rmse[:, i], c = '#0000ff', label = 'Test')
            axes[i].legend()
            axes[i].set_xlabel('Reliability')
            axes[i].set_ylabel(f'RMSE ({u})')
            axes[i].set_title(display_name)
        
        fig.suptitle('Reliability vs RMSE')
        plt.savefig(f'./{args.model}_models/rmse_vs_reliability.png')
        plt.close()

    elif args.mode == 'metrics':
        train_metric_analysis = pd.DataFrame()
        for st in train_stations:
            for ind, model in zip([0, 2], ['Model', 'CMAQ']):
                for i, sp in enumerate(['FSPMC', 'O3']):
                    for metric in [MAE, RMSE, R]:
                        train_metric_analysis[st, model, sp, metric.__name__] = metric(train_pred[st][ind], train_pred[st][1])[i, :]
        train_metric_analysis.to_csv(f'./{args.model}_models/train_metric_analysis.csv')

        test_metric_analysis = pd.DataFrame()
        for st in test_stations:
            for ind, model in zip([0, 2], ['Model', 'CMAQ']):
                for i, sp in enumerate(['FSPMC', 'O3']):
                    for metric in [MAE, RMSE, R]:
                        test_metric_analysis[st, model, sp, metric.__name__] = metric(test_pred[st][ind], test_pred[st][1])[i, :]
    
        # train_metric_analysis.to_csv(f'./{args.model}_models/train_metric_analysis.csv')
        test_metric_analysis.to_csv(f'./{args.model}_models/test_metric_analysis.csv')