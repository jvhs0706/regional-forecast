import torch
import numpy as np
from datetime import date
import pandas as pd
import random

import pickle as pk
from dataset_utils.dataset_source import SourceObservationDataset
from dataset_utils.dataset_target import TargetObservationDataset, TargetWrfCmaqDataset

from geopy.distance import geodesic

class RegionalDataset(torch.utils.data.Dataset):
    def __init__(self, target_stations: list):
        super().__init__()
        self.source = SourceObservationDataset()
        self.target_wrf_cmaq = TargetWrfCmaqDataset(target_stations)
        self.target_observation = TargetObservationDataset(target_stations)

        with open('./data/lat_lon_source.pkl', 'rb') as f:
            source_loc = pk.load(f)
            self.source_stations = list(source_loc.keys())
        with open('./data/lat_lon_target.pkl', 'rb') as f:
            target_loc = pk.load(f)
        
        self.target_stations = target_stations
        self.dist = torch.tensor([
            [geodesic(source_loc[_s], target_loc[_t]).km for _t in self.target_stations] for _s in self.source_stations
        ])
        
        assert len(self.source) == len(self.target_wrf_cmaq) == len(self.target_observation)
        self.length = len(self.source)

    def __len__(self):
        return self.length

    def __getitem__(self, index:int):
        return self.source[index], self.target_wrf_cmaq[index], self.target_observation[index]



if __name__ == '__main__':
    pass