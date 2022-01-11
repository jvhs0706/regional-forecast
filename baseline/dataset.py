import torch
import numpy as np
import pandas as pd

import pickle as pk

history, horizon = 3, 2
target_species = ['AQ_FSPMC', 'AQ_O3']
target_species_cmaq = ['FSPMC', 'O3']

def _wind_conversion(df: pd.DataFrame):
    index = df.index
    magnitude, direction = df['A_WIND_S'].values, df['A_WIND_D'].values 
    x, y = magnitude * np.cos((np.pi/180) * direction), magnitude * np.sin((np.pi/180) * direction)
    df = df.drop(columns=['A_WIND_S', 'A_WIND_D'])
    df['A_WIND_X'], df['A_WIND_Y'] = x, y
    df.set_index(index)
    return df

class _BaselineTimeSeriesNormalizer:
    '''
    Input shape: [C, T]
    parameters: mean and std
    '''
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        assert mean.size == std.size
        mean, std = mean.reshape(mean.size, 1), std.reshape(std.size, 1)
        self.mean, self.std = torch.tensor(mean), torch.tensor(std)

    def __call__(self, x):
        return (x - self.mean) / self.std

class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, station):
        super().__init__()
        self.history, self.horizon = history, horizon
        
        with open('../data/obs_data_source.pkl', 'rb') as f:
            obs = pk.load(f)[station]
            obs = _wind_conversion(obs)
        self.obs_features, self.obs = list(obs.columns), obs.values
        assert self.obs.shape[0] % 24 == 0
        self.obs_normalizer = _BaselineTimeSeriesNormalizer(np.nanmean(self.obs, axis = 0), np.nanstd(self.obs, axis = 0))
        self.target_index = [self.obs_features.index(sp) for sp in target_species]

        with open('../data/wrf_data_target.pkl', 'rb') as f:
            wrf = pk.load(f)[station]
        with open('../data/cmaq_data_target.pkl', 'rb') as f:
            cmaq = pk.load(f)[station]

        self.wrf_cmaq_features = []
        self.wrf_cmaq = []
        for sp, arr in {**wrf, **cmaq}.items():
            self.wrf_cmaq_features.append(sp)
            self.wrf_cmaq.append(arr)
        self.wrf_cmaq = np.stack(self.wrf_cmaq, axis = 1) # [n_days, n_features, T]
        assert 24 * self.horizon == self.wrf_cmaq.shape[2]
        self.wrf_cmaq_normalizer =\
            _BaselineTimeSeriesNormalizer(np.mean(self.wrf_cmaq, axis = (0, 2)), np.std(self.wrf_cmaq, axis = (0, 2)))

    def __len__(self):
        return self.obs.shape[0] // 24 - (self.history + self.horizon) + 1

    def __getitem__(self, index:int):
        X0 = torch.tensor(self.obs[24*index:24*(index+self.history)].T)
        X0 = self.obs_normalizer(X0)
        X0 = torch.nan_to_num(X0).float()

        X1 = self.wrf_cmaq_normalizer(torch.tensor(self.wrf_cmaq[index+self.history]))

        y = torch.tensor(self.obs[24*(index+self.history): 24*(index+self.history+self.horizon), self.target_index].T)
        return X0, X1, y

class BaselineTargetDataset(torch.utils.data.Dataset):
    def __init__(self, station):
        super().__init__()
        self.history, self.horizon = history, horizon
        
        with open('../data/obs_data_target.pkl', 'rb') as f:
            obs = pk.load(f)[station]
            self.obs_features, self.obs = list(obs.columns), obs.values
        assert self.obs.shape[0] % 24 == 0

        with open('../data/cmaq_data_target.pkl', 'rb') as f:
            cmaq = pk.load(f)[station]

        self.cmaq_features, self.cmaq= [], []
        for sp in target_species_cmaq:
            self.cmaq_features.append(sp)
            self.cmaq.append(cmaq[sp])
        self.cmaq = np.stack(self.cmaq, axis = 1) # [n_days, n_features, T]
        assert 24 * self.horizon == self.cmaq.shape[2]

    def __len__(self):
        return self.obs.shape[0] // 24 - (self.history + self.horizon) + 1

    def __getitem__(self, index:int):
        cmaq = torch.tensor(self.cmaq[index + self.history])
        y = torch.tensor(self.obs[24*(index+self.history): 24*(index+self.history+self.horizon)].T)
        return cmaq, y