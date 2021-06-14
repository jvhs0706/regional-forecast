import torch
import numpy as np
from datetime import date
import pandas as pd
import pickle as pk

from .utils import *

def _process_wrf_cmaq_dict(dic: dict):
    species = list(dic.keys())
    data = np.stack([arr for arr in dic.values()], axis = -1)
    return species, data

def _wind_conversion(df: pd.DataFrame):
    index = df.index
    magnitude, direction = df['A_WIND_S'].values, df['A_WIND_D'].values 
    x, y = magnitude * np.cos((np.pi/180) * direction), magnitude * np.sin((np.pi/180) * direction)
    df = df.drop(columns=['A_WIND_S', 'A_WIND_D'])
    df['A_WIND_X'], df['A_WIND_Y'] = x, y
    df.set_index(index)
    return df

class Single_Station_SourceObservationDataset(torch.utils.data.Dataset):
    def __init__(self, obs:pd.DataFrame, history: int = history_nday, horizon: int = horizon_nday):
        '''
        obs: [24 * num_days, num_features]
        '''
        
        super().__init__()
        obs = _wind_conversion(obs)
        self.features, self.obs = list(obs.columns), obs.values
        self.history, self.horizon = history, horizon
        
        assert self.obs.shape[0] % 24 == 0

        self.normalizer = TimeSeriesNormalizer(np.nanmean(self.obs, axis = 0), np.nanstd(self.obs, axis = 0))

    def __len__(self):
        return self.obs.shape[0] // 24 - (self.history + self.horizon) + 1

    def __getitem__(self, index:int):
        X = torch.tensor(self.obs[24*index:24*(index+self.history)].T)
        X = self.normalizer(X)
        X = torch.nan_to_num(X).float()
        return X

class SourceObservationDataset(torch.utils.data.Dataset):
    def __init__(self, obs_fn: str = './data/obs_data_source.pkl'):
        super().__init__()
        with open(obs_fn, 'rb') as f:
            obs = pk.load(f)
        
        self.station_datasets = {}
        self.length = None
        for st, df in obs.items():
            self.station_datasets[st] = Single_Station_SourceObservationDataset(df)
            if self.length is None:
                self.length = len(self.station_datasets[st])
            else:
                assert self.length == len(self.station_datasets[st])

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        return {st: ds[index] for st, ds in self.station_datasets.items()}


