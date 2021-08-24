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
from data.match import *

from datetime import datetime, timedelta

def MAE(pred, y, axis = 0):
    return np.nanmean(np.abs(pred - y), axis = axis)

def RMSE(pred, y, axis = 0):
    return np.sqrt(np.nanmean((pred - y)**2, axis = axis))

def SMAPE(pred, y, axis = 0):
    return 200 * np.nanmean(np.abs(pred - y) / (np.abs(pred) + np.abs(y)), axis = axis)

def R(pred, y, axis = 0):
    mask = ~np.isnan(y)
    n = mask.sum(axis = axis, keepdims = True)
    pred_mean = np.sum(pred * mask, axis = axis, keepdims = True) / n
    y_mean = np.nanmean(y, axis = axis, keepdims = True)
    r = np.nansum((pred - pred_mean) * (y - y_mean), axis = axis)/ np.sqrt(np.sum(mask * (pred - pred_mean)**2, axis = axis) * np.nansum((y - y_mean)**2, axis = axis))
    return r

metrics = [MAE, RMSE, SMAPE, R]

def regional_reliability(source_lat_lon, weight_decay, lat_range, lon_range):
    x, y = np.linspace(*lon_range, 100), np.linspace(*lat_range, 100)
    X, Y = np.meshgrid(x, y)
    out = np.zeros_like(X)
    for decay_factor, (lat, lon) in zip(weight_decay, source_lat_lon.values()):
        dist = np.apply_along_axis(lambda z: geodesic((z[1], z[0]), (lat, lon)).km, axis = 0, arr = np.stack([X, Y]))
        out += np.exp(-decay_factor * dist)
    return X, Y, out        

def plot_ax(st, sp, u, ax, time_axis, obs, pred, time_lag_day):
    ax.plot(time_axis, obs, 'ro', label = 'Observation', markersize = 0.5)
    ax.plot(time_axis, pred, 'b', label = 'Prediction', linewidth = 0.5)
    ax.legend(fontsize = 20)
    ax.set_title(f'{sp} ({u}), time-lags {24 * time_lag_day} - {24 * time_lag_day + 23} h', fontsize = 24)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = 'Model name.')
    parser.add_argument('mode', choices= ['overall', 'temporal', 'metrics', 'plots'])
    parser.add_argument('-bs', '--batch_size', type = int, help = 'Batch size for loading data.', default = 64)
    parser.add_argument('--baseline', action = 'store_true', help = 'Load baseline data.')
    args = parser.parse_args()
    
    # model loading
    model = torch.load(f'./{args.model}_models/model.nctmo')
    model.eval()
    decay = model.gcn.distance_decay.detach().numpy()
    
    with open('./data/lat_lon_source.pkl', 'rb') as f:
        source_lat_lon = pk.load(f)
        source_stations = list(source_lat_lon.keys())
    
    with open(f'./{args.model}_models/train.txt', 'r') as f:
        train_stations = f.read().splitlines()
    with open(f'./{args.model}_models/test.txt', 'r') as f:
        test_stations = f.read().splitlines()
        if args.baseline:
            test_stations = [st for st in test_stations if st not in source_stations]     
    
    with open('./data/lat_lon_target.pkl', 'rb') as f:
        target_lat_lon = pk.load(f) 
    
    with open(f'./{args.model}_models/source_normalizers.pkl', 'rb') as f:
        source_normalizers = pk.load(f)
    with open(f'./{args.model}_models/wrf_cmaq_normalizers.pkl', 'rb') as f:
        wrf_cmaq_normalizers = pk.load(f)
    
    with open(f'./{args.model}_models/temporal_split.pkl', 'rb') as f:
        split = pk.load(f)
        train_dates, test_dates = split['train'], split['test']

    if args.baseline:
        with open(f'./{args.model}_models/baseline/baseline_predictions.pkl', 'rb') as f:
            baseline_pred = pk.load(f)


    test = RegionalDataset(test_stations)
    test_subset = torch.utils.data.Subset(test, test_dates)
    test.target_wrf_cmaq.normalizer = wrf_cmaq_normalizers
    for st in source_stations:
        test.source.station_datasets[st].normalizer = source_normalizers[st]
    
    test_dataloader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
    test_pred, cmaq_pred, ground_truth = {st: [] for st in test_stations}, {}, {st: [] for st in test_stations}
    test_target_features = [test.target_wrf_cmaq.features.index('FSPMC'), test.target_wrf_cmaq.features.index('O3')]
    with torch.no_grad():
        for j, (X0, X1, obs) in enumerate(test_dataloader):
            pred = model(X0, X1, test.dist).detach()
            for k, st in enumerate(test_stations):
                test_pred[st].append(pred[k])
                ground_truth[st].append(obs[st].detach())
            print(f'Testing on test set, batch {j}...')
    test_pred = {st: np.concatenate(arr) for st, arr in test_pred.items()}
    ground_truth = {st: np.concatenate(arr) for st, arr in ground_truth.items()}

    for st in test_stations:
        ds = test.target_wrf_cmaq.station_datasets[st]
        cmaq_pred[st] = np.stack(
                [
                    ds.wrf_cmaq[ds.history: ds.history + len(ds), test_target_features[0], :],
                    1000 * ds.wrf_cmaq[ds.history: ds.history + len(ds), test_target_features[1], :]
                ], axis = 1
            )[test_dates]

    if args.mode == 'overall':
        pred, y, cmaq = np.concatenate([test_pred[st] for st in test_stations], axis = 0),\
            np.concatenate([ground_truth[st] for st in test_stations], axis = 0), np.concatenate([cmaq_pred[st] for st in test_stations], axis = 0)
        assert y.shape == pred.shape == cmaq.shape
        N, _, _ = y.shape
        y, pred, cmaq = y.reshape(N, 2, 2, 24), pred.reshape(N, 2, 2, 24), cmaq.reshape(N, 2, 2, 24)
        
        if args.baseline:
            bl = np.concatenate([baseline_pred[st] for st in test_stations], axis = 0)
            bl = bl.reshape(N, 2, 2, 24)
            assert y.shape == bl.shape, (y.shape, bl.shape)
        
        for m in metrics:
            cmaq_values = m(pred = cmaq, y = y, axis = (0, 3))
            if args.baseline:
                baseline_values = m(pred = bl, y = y, axis = (0, 3))
            values = m(pred = pred, y = y, axis = (0, 3))
            
            for i, sp in enumerate(['FSPMC', 'O3']):
                for j in range(2):
                    print(f'CMAQ, Species {sp}, day {j}, metric {m.__name__}: {cmaq_values[i, j]:.2f}')
                    if args.baseline:
                        print(f'Baseline, Species {sp}, day{j}, metric {m.__name__}: {baseline_values[i, j]:.2f}')
                    print(f'Model, Species {sp}, day {j}, metric {m.__name__}: {values[i, j]:.2f}')

    elif args.mode == 'temporal':
        pred, y, cmaq = np.concatenate([test_pred[st] for st in test_stations], axis = 0),\
            np.concatenate([ground_truth[st] for st in test_stations], axis = 0), np.concatenate([cmaq_pred[st] for st in test_stations], axis = 0)
        assert y.shape == pred.shape == cmaq.shape
        N, _, _ = y.shape
        
        if args.baseline:
            bl = np.concatenate([baseline_pred[st] for st in test_stations], axis = 0)
            assert y.shape == bl.shape, (y.shape, bl.shape)

        df = pd.DataFrame()
        for m in metrics:
            values = m(pred = pred, y = y, axis = 0)
            if args.baseline:
                baseline_values = m(pred = bl, y = y, axis = 0)
            cmaq_values = m(pred = cmaq, y = y, axis = 0)
            for i, sp in enumerate(['FSPMC', 'O3']):
                df[m.__name__, 'Model', sp], df[m.__name__, 'CMAQ', sp] = values[i], cmaq_values[i]
                if args.baseline:
                    df[m.__name__, 'Spatial correction', sp] = baseline_values[i]
        
        for i, (sp, sp_name, unit) in enumerate(zip(['FSPMC', 'O3'], ['$\mathrm{PM}_{2.5}$', '$\mathrm{O}_3$'], ['$\mu g/m^3$', 'ppbv'])):
            fig, axes = plt.subplots(len(metrics), 1, figsize = (16, 24))
            for j, (m, u) in enumerate(zip(metrics, [unit, unit, '%', None])):
                axes[j].plot(df[m.__name__, 'Model', sp], label = 'Model')
                if args.baseline:
                    axes[j].plot(df[m.__name__, 'Spatial correction', sp], label = 'Spatial correction')
                axes[j].plot(df[m.__name__, 'CMAQ', sp], label = 'CMAQ')
                axes[j].legend(fontsize = 16)
                axes[j].set_title(f'{m.__name__} ({u})' if u is not None else f'{m.__name__}', fontsize = 24)
                axes[j].xaxis.set_tick_params(labelsize=16)
                axes[j].yaxis.set_tick_params(labelsize=16)
                axes[j].set_xlabel('Time-lag', fontsize=20)
            plt.tight_layout()
            plt.savefig(f'./{args.model}_models/temporal_{sp}.png')
            plt.close()
                

        assert not args.baseline
        plt.contourf(*regional_reliability(source_lat_lon, decay, (lat_min, lat_max), (lon_min, lon_max)))
        plt.colorbar()
        plt.title('Reliability')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.plot([loc[1] for loc in source_lat_lon.values()], [loc[0] for loc in source_lat_lon.values()], 'or', markersize = 3, label = 'Source stations')
        borders = get_border()
        for (blon, blat) in borders:
            plt.plot(blon, blat, 'k', linewidth=1)
        plt.legend()
        plt.savefig(f'./{args.model}_models/reliability_map.png')
        plt.close()
        

        test_metrics = {m.__name__: np.stack([m(test_pred[st].reshape(-1, 2, 2, 24), ground_truth[st].reshape(-1, 2, 2, 24), axis = (0, 3)) for st in test_stations]) for m in metrics}
            
        test_reliability = np.exp(-decay[:, None] * test.dist.detach().numpy()).sum(axis = 0)
        
        
        fig, axes = plt.subplots(4, 4, figsize = (50, 50))
        source_mask = np.array([st in source_stations for st in test_stations])
        for j, m in enumerate(metrics):
            for i, (sp, display_name, unit) in enumerate(zip(['fspmc', 'o3'], ['$\mathrm{PM}_{2.5}$', '$\mathrm{O}_3$'], ['$\mu g/m^3$', 'ppbv'])):
                axes[j, 2 * i + 1].sharey(axes[j, 2 * i])
                for t in range(2):
                    axes[j, 2 * i + t].scatter(test_reliability[source_mask], test_metrics[m.__name__][source_mask, i, t], c = '#ff0000', s = 50, label = 'Source')
                    axes[j, 2 * i + t].scatter(test_reliability[~source_mask], test_metrics[m.__name__][~source_mask, i, t], c = '#0000ff', s = 50, label = 'Non-source')
                    axes[j, 2 * i + t].legend(fontsize = 24)
                    axes[j, 2 * i + t].set_xlabel('Reliability', fontsize = 28)
                    axes[j, 2 * i + t].xaxis.set_tick_params(labelsize=24)
                    axes[j, 2 * i + t].yaxis.set_tick_params(labelsize=24)
                    u = f' ({unit})' if j < 2 else ' (%)' if j == 2 else ''

                    axes[j, 2 * i + t].set_title(f'{display_name} {m.__name__}{u}, time-lags {24 * t} - {24 * t + 23} h', fontsize = 32)
        plt.tight_layout()
        plt.savefig(f'./{args.model}_models/metrics_reliability.png')
        plt.close()

    elif args.mode == 'metrics':
        raise NotImplementedError
        # train_metric_analysis = pd.DataFrame()
        # for st in train_stations:
        #     for ind, model in zip([0, 2], ['Model', 'CMAQ']):
        #         for i, sp in enumerate(['FSPMC', 'O3']):
        #             for metric in [MAE, RMSE, R]:
        #                 train_metric_analysis[st, model, sp, metric.__name__] = metric(train_pred[st][ind], train_pred[st][1])[i, :]
        # train_metric_analysis.to_csv(f'./{args.model}_models/train_metric_analysis.csv')

        # test_metric_analysis = pd.DataFrame()
        # for st in test_stations:
        #     for ind, model in zip([0, 2], ['Model', 'CMAQ']):
        #         for i, sp in enumerate(['FSPMC', 'O3']):
        #             for metric in [MAE, RMSE, R]:
        #                 test_metric_analysis[st, model, sp, metric.__name__] = metric(test_pred[st][ind], test_pred[st][1])[i, :]
    
        # # train_metric_analysis.to_csv(f'./{args.model}_models/train_metric_analysis.csv')
        # test_metric_analysis.to_csv(f'./{args.model}_models/test_metric_analysis.csv')

    elif args.mode == 'plots':
        assert not args.baseline
        time_axis = [datetime(2020, 1, 2, 9) + timedelta(hours = h) for h in range(364 * 24)]
        if not os.path.isdir(f'./{args.model}_models/plots'):
            os.makedirs(f'./{args.model}_models/plots')
        for st in test_stations:
            lat, lon = target_lat_lon[st]
            fig, axes = plt.subplots(2, 2, figsize = (30, 20))
            fig.suptitle(f'Station {st} ({lat:.2f}°N, {lon:.2f}°E)', fontsize = 36)
            for i, (sp, display_name, unit) in enumerate(zip(['FSPMC', 'O3'], ['$\mathrm{PM}_{2.5}$', '$\mathrm{O}_3$'], ['$\mu g/m^3$', 'ppbv'])):
                obs = ground_truth[st][1:, i, :24].flatten()
                pred = [test_pred[st][1:, i, :24].flatten(), test_pred[st][:-1, i, 24:].flatten()]
                axes[i, 1].sharey(axes[i, 0])
                for j in range(2):
                    plot_ax(st, display_name, unit, axes[i, j], time_axis, obs, pred[j], j)
            plt.savefig(f'./{args.model}_models/plots/{st}.png')
            plt.close()