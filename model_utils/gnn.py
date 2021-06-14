import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .utils import *
from .lstm import *

class BipartiteLSTMGCN1D(nn.Module):
    def __init__(self, num_channels:list, num_source: int, bn = True):
        '''
        channels_left: [in_channels, hidden_channels1, hidden_channels2, ..., out_channels]
        channels_right: [in_channels, hidden_channels1, hidden_channels2, ..., out_channels]
        '''
        super().__init__()
        self.distance_decay = nn.Parameter(torch.zeros(num_source))
        self.lstm = nn.Sequential(*[BidirectionalLSTM(_in, _out, bn) for _in, _out in zip(num_channels[:-1], num_channels[1:])])
    
    def forward(self, X: torch.dist, dist: torch.dist):
        (U, V), (U, N, C, T) = dist.shape, X.shape
        weight = torch.exp(-self.distance_decay[:, None] * dist)
        weight = weight / torch.sum(weight, axis = 0)
        out = self.lstm(X.view(U*N, C, T)) 
        out = out.view(U, N, -1, T).permute(1, 2, 3, 0)
        out = torch.matmul(out, weight).permute(3, 0, 1, 2)
        return out
