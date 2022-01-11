from model import *
from dataset import *
from metrics import *
from data.match import *
from model import Regional
from dataset import RegionalDataset
from plot_stations import get_border

import pickle as pk
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import pandas as pd
from datetime import datetime, timedelta, date

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    parser.add_argument('mode', choices= ['overall', 'temporal'])
    parser.add_argument('-bs', '--batch_size', type = int, help = 'Batch size for loading data.', default = 64)
    parser.add_argument('-bl', '--baselines', nargs = '*', help = 'Load baseline data.')
    args = parser.parse_args()
    
    # model loading
    model = torch.load(f'./{args.model}_models/model.nctmo')
    model.eval()
    
    with open('./data/lat_lon_source.pkl', 'rb') as f:
        source_lat_lon = pk.load(f)
        source_stations = list(source_lat_lon.keys())
    
    with open(f'./{args.model}_models/train.txt', 'r') as f:
        train_stations = f.read().splitlines()
    with open(f'./{args.model}_models/test.txt', 'r') as f:
        test_stations = f.read().splitlines()
        if len(args.baselines) > 0:
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

    test = RegionalDataset(test_stations)
    test_subset = torch.utils.data.Subset(test, test_dates)
    test.target_wrf_cmaq.normalizer = wrf_cmaq_normalizers
    for st in source_stations:
        test.source.station_datasets[st].normalizer = source_normalizers[st]
    
    # load the test predictions, cmaq predicions, and the ground truth
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

    # If needed, load the baseline predictions
    if len(args.baselines) > 0:
        baseline_pred = {}
        for bl in args.baselines:
            with open(f'./{args.model}_models/baseline/baseline_{bl}_predictions.pkl', 'rb') as f:
                baseline_pred[bl] = pk.load(f)

    pred, y, cmaq = np.concatenate([test_pred[st] for st in test_stations], axis = 0),\
        np.concatenate([ground_truth[st] for st in test_stations], axis = 0), np.concatenate([cmaq_pred[st] for st in test_stations], axis = 0)
    assert y.shape == pred.shape == cmaq.shape
    N, _, _ = y.shape
    y, pred, cmaq = y.reshape(N, 2, 2, 24), pred.reshape(N, 2, 2, 24), cmaq.reshape(N, 2, 2, 24)
    
    if len(args.baselines) > 0:
        baseline = {bl: np.concatenate([baseline_pred[bl][st] for st in test_stations], axis = 0).reshape(N, 2, 2, 24) for bl in args.baselines}
    
    for m in metrics:
        cmaq_values = m(pred = cmaq, y = y, axis = (0, 3))
        if len(args.baselines) > 0:
            baseline_values = {bl: m(pred = baseline[bl], y = y, axis = (0, 3)) for bl in args.baselines}
        values = m(pred = pred, y = y, axis = (0, 3))
        
        for i, sp in enumerate(['FSPMC', 'O3']):
            for j in range(2):
                print(f'CMAQ, Species {sp}, day {j}, metric {m.__name__}: {cmaq_values[i, j]:.4g}')
                if len(args.baselines) > 0:
                    for bl in args.baselines:
                        print(f'Baseline {bl}, Species {sp}, day{j}, metric {m.__name__}: {baseline_values[bl][i, j]:.4g}')
                print(f'Broadcasting, Species {sp}, day {j}, metric {m.__name__}: {values[i, j]:.4g}')

    pred, y, cmaq = np.concatenate([test_pred[st] for st in test_stations], axis = 0),\
        np.concatenate([ground_truth[st] for st in test_stations], axis = 0), np.concatenate([cmaq_pred[st] for st in test_stations], axis = 0)
    assert y.shape == pred.shape == cmaq.shape
    N, _, _ = y.shape
    
    if len(args.baselines) > 0:
        baseline = {bl: np.concatenate([baseline_pred[bl][st] for st in test_stations], axis = 0) for bl in args.baselines}

    df = pd.DataFrame()
    for m in metrics:
        values = m(pred = pred, y = y, axis = 0)
        if len(args.baselines) > 0:
            baseline_values_dic = {bl: m(pred = baseline[bl], y = y, axis = 0) for bl in args.baselines}
            extremum_func = np.max if m.__name__ == 'R' else np.min
            baseline_values = extremum_func(np.stack(list(baseline_values_dic.values())), axis = 0)
        cmaq_values = m(pred = cmaq, y = y, axis = 0)
        for i, sp in enumerate(['FSPMC', 'O3']):
            df[m.__name__, 'Broadcasting', sp], df[m.__name__, 'CMAQ', sp] = values[i], cmaq_values[i]
            if len(args.baselines) > 0:
                df[m.__name__, 'Interpolation', sp] = baseline_values[i]
    
    fig = plt.Figure(figsize = (30, 30), constrained_layout=True)
    cols = fig.subfigures(1, 2)
    for i, (sp, sp_name, unit) in enumerate(zip(['FSPMC', 'O3'], ['$\mathrm{PM}_{2.5}$', '$\mathrm{O}_3$'], ['$\mu g/m^3$', 'ppbv'])):
        cols[i].suptitle(f'{sp_name}', fontsize = 30)
        axes = cols[i].subplots(4, 1)
        for j, (m, u) in enumerate(zip(metrics, [unit, unit, '%', None])):
            axes[j].plot(df[m.__name__, 'Broadcasting', sp], label = 'Broadcasting')
            if len(args.baselines) > 0:
                axes[j].plot(df[m.__name__, 'Interpolation', sp], label = 'Interpolation')
            axes[j].plot(df[m.__name__, 'CMAQ', sp], label = 'CMAQ')
            axes[j].legend(fontsize = 16)
            axes[j].set_title(f'{m.__name__} ({u})' if u is not None else f'{m.__name__}', fontsize = 24)
            axes[j].xaxis.set_tick_params(labelsize=16)
            axes[j].yaxis.set_tick_params(labelsize=16)
            axes[j].set_xlabel('Time-lag (h)', fontsize=20)
    fig.savefig(f'./{args.model}_models/temporal.png')
    plt.close(fig)