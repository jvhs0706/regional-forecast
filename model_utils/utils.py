import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class _TimeDistributedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_step: int):
        super().__init__()
        self.in_features, self.out_features, self.num_step = in_features, out_features, num_step
        weight, bias = torch.empty((num_step, out_features, in_features)), torch.empty((num_step, out_features))
        self.weight, self.bias = nn.Parameter(weight), nn.Parameter(bias)
        nn.init.uniform_(self.weight, -np.sqrt(6/(in_features+out_features)), +np.sqrt(6/(in_features+out_features)))
        nn.init.constant_(self.bias, 0)
    def forward(self, _x):
        N, T, C = _x.shape 
        assert (C, T) == (self.in_features, self.num_step)
        out = torch.matmul(_x[:, :, None, :], self.weight.transpose(1, 2)[None, :]).squeeze(2)
        out += self.bias[None, :, :]
        return out 

class Dense1D(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_step: int, dropout: float, time_distributed: bool, activation = F.relu):
        super().__init__()
        if dropout > 0.0:
            self.dropout = nn.Dropout(p = dropout)
        self.linear = _TimeDistributedLinear(in_features, out_features, num_step) if time_distributed else nn.Linear(in_features, out_features)
        if activation is not None:
            self.activation = activation
        
    def forward(self, x):
        N, C, T = x.shape 
        assert C == self.linear.in_features, f'{self.linear.in_features}, {C}'
        out = self.dropout(x).transpose(1, 2) if hasattr(self, 'dropout') else x.transpose(1, 2)
        out = self.linear(out)
        out = self.activation(out) if hasattr(self, 'activation') else out
        return out.transpose(1, 2)