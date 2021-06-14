import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .utils import *

class LSTMEncoder(nn.Module):
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

class LSTMDecoder(nn.Module):
    def __init__(self, encoded_size: int, length: int, bn: bool):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(0, encoded_size)
        self.length = length 
        if bn:
            self.bn = nn.BatchNorm1d(encoded_size)

    def forward(self, h_in):
        N, H = h_in.shape
        assert H == self.lstm_cell.hidden_size
        out, c_in = [], torch.zeros((N, H))
        
        for t in range(self.length):
            if t == 0:
                h, c = self.lstm_cell(torch.zeros((N, 0)), (h_in, c_in))
            else:
                h, c = self.lstm_cell(torch.zeros((N, 0)), (h, c))
            out.append(h)

        out = torch.stack(out, axis = -1)
        if hasattr(self, 'bn'):
            out = self.bn(out)
        return out



class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, output_size: int, bn: bool):
        super().__init__()
        assert output_size % 2 == 0
        self.flstm_cell = nn.LSTMCell(input_size, output_size // 2)
        self.blstm_cell = nn.LSTMCell(input_size, output_size // 2)
        if bn:
            self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        N, C, T = x.shape 
        H, H = self.flstm_cell.hidden_size, self.blstm_cell.hidden_size
        fout, bout = [], []
    
        for t in range(T):
            if t == 0:
                h, c = self.flstm_cell(x[:, :, t])
            else:
                h, c = self.flstm_cell(x[:, :, t], (h, c))
            fout.append(h)
    
        for t in reversed(range(T)):
            if t == T-1:
                h, c = self.flstm_cell(x[:, :, t])
            else:
                h, c = self.flstm_cell(x[:, :, t], (h, c))
            bout.append(h)
        
        out = torch.cat([torch.stack(fout, axis = -1), torch.stack(list(reversed(bout)), axis = -1)], axis = 1)
        if hasattr(self, 'bn'):
            out = self.bn(out)
        return out


class EncoderDecoder(nn.Module):
    def __init__(self, in_channels: int, encoded_size: int, num_step: int, out_channels: int, dropout: float, bn: bool):
        super().__init__()
        self.encoder = LSTMEncoder(in_channels = in_channels, encoded_size = encoded_size)
        self.decoder = LSTMDecoder(encoded_size = encoded_size, length = num_step, bn = bn)
        self.dense = Dense1D(in_features = encoded_size, out_features = out_channels, num_step = num_step, dropout = dropout, time_distributed = False)
    
    def forward(self, x):
        out = self.dense(self.decoder(self.encoder(x)))
        return out