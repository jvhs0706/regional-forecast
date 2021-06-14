import torch
import numpy as np
from datetime import date
import pandas as pd
import pickle as pk

from .utils import *

class Single_Station_TargetObservationDataset(torch.utils.data.Dataset):
    def __init__(self, obs:pd.DataFrame, history: int = history_nday, horizon: int = horizon_nday):
        '''
        obs: [24 * num_days, num_features]
        '''
        super().__init__()
        self.features, self.obs = list(obs.columns), obs.values
        self.history, self.horizon = history, horizon
        
        assert self.obs.shape[0] % 24 == 0

    def __len__(self):
        return self.obs.shape[0] // 24 - (self.history + self.horizon) + 1

    def __getitem__(self, index:int):
        y = torch.tensor(self.obs[24*(index+self.history):24*(index+self.history+self.horizon)].T)
        return y

class TargetObservationDataset(torch.utils.data.Dataset):
    def __init__(self, stations: list, obs_fn: str = './data/obs_data_target.pkl'):
        super().__init__()
        with open(obs_fn, 'rb') as f:
            obs = pk.load(f)
        
        self.station_datasets = {}
        self.length = None
        self.features = None
        for st in stations:
            df = obs[st]
            self.station_datasets[st] = Single_Station_TargetObservationDataset(df)
            if self.length is None and self.features is None:
                self.length = len(self.station_datasets[st])
                self.features = self.station_datasets[st].features
            else:
                assert self.length == len(self.station_datasets[st])
                assert self.features == self.station_datasets[st].features

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        return {st: ds[index] for st, ds in self.station_datasets.items()}


class Single_Station_TargetWrfCmaqDataset(torch.utils.data.Dataset):
    def __init__(self, wrf: dict, cmaq: dict, history: int = history_nday, horizon: int = horizon_nday):
        super().__init__()
        self.features = []
        self.wrf_cmaq = []
        for sp, arr in {**wrf, **cmaq}.items():
            self.features.append(sp)
            self.wrf_cmaq.append(arr)
        self.wrf_cmaq = np.stack(self.wrf_cmaq, axis = 1) # [n_days, n_features, T]
        self.history, self.horizon = history, horizon
        assert 24 * self.horizon == self.wrf_cmaq.shape[2]

    def __len__(self):
        return self.wrf_cmaq.shape[0] - (self.history + self.horizon) + 1

    def __getitem__(self, index:int):
        return torch.tensor(self.wrf_cmaq[index+self.history])

class TargetWrfCmaqDataset(torch.utils.data.Dataset):
    def __init__(self, stations: list, wrf_fn: str = './data/wrf_data_target.pkl', cmaq_fn: str = './data/cmaq_data_target.pkl'):
        super().__init__()
        with open(wrf_fn, 'rb') as f:
            wrf = pk.load(f)
        with open(cmaq_fn, 'rb') as f:
            cmaq = pk.load(f)
        
        self.station_datasets = {}
        self.length = None
        self.features = None
        for st in stations:
            self.station_datasets[st] = Single_Station_TargetWrfCmaqDataset(wrf[st], cmaq[st])
            if self.length is None and self.features is None:
                self.length = len(self.station_datasets[st])
                self.features = self.station_datasets[st].features
            else:
                assert self.length == len(self.station_datasets[st])
                assert self.features == self.station_datasets[st].features
        self.normalizer = TimeSeriesNormalizer(
            mean = np.concatenate([ds.wrf_cmaq for ds in self.station_datasets.values()], axis = 0).mean(axis = (0, 2)), 
            std = np.concatenate([ds.wrf_cmaq for ds in self.station_datasets.values()], axis = 0).std(axis = (0, 2))
        ) 
        

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        return {st: self.normalizer(ds[index]) for st, ds in self.station_datasets.items()}