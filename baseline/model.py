import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class _BaselineLSTMEncoder(nn.Module):
    def __init__(self, in_channels: int, encoded_size: int):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(in_channels, encoded_size)

    def forward(self, x):
        N, C, T = x.shape 
        assert C == self.lstm_cell.input_size
        C, H = self.lstm_cell.input_size, self.lstm_cell.hidden_size
        out, c = torch.zeros((N, H)), torch.zeros((N, H))
        for t in range(T):
            out, c = self.lstm_cell(x[:, :, t], (out, c))
        return out

class _BaselineDense(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation = F.relu, dropout = 0.35):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)
        self.linear = nn.Linear(in_channels, out_channels)
        if activation is not None:
            self.activation = activation

    def forward(self, x): 
        out = self.linear(self.dropout(x)) 
        if hasattr(self, 'activation'):
            out = self.activation(out)
        return out

class Baseline(nn.Module):
    
    def __init__(self, in_channels: int, wrf_cmaq_in_channels: int):
        super().__init__()
        self.obs_input = _BaselineLSTMEncoder(in_channels, 64)
        self.wrf_cmaq_input = _BaselineLSTMEncoder(wrf_cmaq_in_channels, 64)
        self.denses_fspmc = nn.Sequential(
            *[_BaselineDense(128, 96), _BaselineDense(96, 64), _BaselineDense(64, 48, activation = None)]
        )
        self.denses_o3 = nn.Sequential(
            *[_BaselineDense(128, 96), _BaselineDense(96, 64), _BaselineDense(64, 48, activation = None)]
        )

    def forward(self, X_obs, X_wrf_cmaq):
        out = torch.cat([self.obs_input(X_obs), self.wrf_cmaq_input(X_wrf_cmaq)], axis = 1)
        out = torch.stack([self.denses_fspmc(out), self.denses_o3(out)], axis = 1)
        return out

        