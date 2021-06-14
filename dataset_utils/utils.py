import torch
import numpy as np

history_nday, horizon_nday = 3, 2
target_species = ['AQ_FSPMC', 'AQ_O3']

class TimeSeriesNormalizer:
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