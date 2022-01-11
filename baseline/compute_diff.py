from model import *
from dataset import *
import pickle as pk
import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from geopy.distance import geodesic

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = 'Model name.')
    parser.add_argument('-bs', '--batch_size', type = int, help = 'Batch size for loading data.', default = 64)
    args = parser.parse_args()
    
    # model loading
    with open(f'../{args.model}_models/train.txt', 'r') as f:
        train_stations = f.read().splitlines()
    with open(f'../{args.model}_models/test.txt', 'r') as f:
        test_stations = f.read().splitlines()
    with open('../data/lat_lon_source.pkl', 'rb') as f:
        source_lat_lon = pk.load(f)
        source_stations = list(source_lat_lon.keys())   
    with open('../data/lat_lon_target.pkl', 'rb') as f:
        target_lat_lon = pk.load(f) 
    with open(f'../{args.model}_models/temporal_split.pkl', 'rb') as f:
        split = pk.load(f)
        train_dates, test_dates = split['train'], split['test']

    baseline_source_predictions = {}
    for st in source_stations:
        print(f'Source station {st}...')
        baseline_dataset = BaselineDataset(st)
        baseline_dataloader = torch.utils.data.DataLoader(baseline_dataset, batch_size=args.batch_size, shuffle=False)
        baseline_model = torch.load(f'../{args.model}_models/baseline/{st}_model.nctmo')
        baseline_model.eval()

        baseline_pred = ([], None)
        for X0, X1, _ in baseline_dataloader:
            baseline_pred[0].append(baseline_model(X0, X1).detach().numpy()) 

        wrf_cmaq_target_indices = [baseline_dataset.wrf_cmaq_features.index('FSPMC'), baseline_dataset.wrf_cmaq_features.index('O3')]
        baseline_pred = \
            (   
                np.concatenate(baseline_pred[0], axis = 0), \
                baseline_dataset.wrf_cmaq[baseline_dataset.history: baseline_dataset.history + len(baseline_dataset), wrf_cmaq_target_indices, :] * np.array([1, 1000]).reshape(1, 2, 1)
            )
        baseline_source_predictions[st] = baseline_pred

    with open(f'../{args.model}_models/baseline/baseline_source_predictions.pkl', 'wb') as f:
        pk.dump(baseline_source_predictions, f)
