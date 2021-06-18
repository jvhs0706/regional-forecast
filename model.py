import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pk

import numpy as np
from model_utils.gnn import BipartiteLSTMGCN1D
from model_utils.lstm import EncoderDecoder, BidirectionalLSTM
from model_utils.utils import Dense1D

class Regional(nn.Module):
    def __init__(self, bn = True, dropout = 0.35):
        super().__init__()
        with open('./data/lat_lon_source.pkl', 'rb') as f:
            source_loc = pk.load(f)
            self.source_stations = list(source_loc.keys())

        with open('./data/obs_data_source.pkl', 'rb') as f:
            obs_source = pk.load(f)
            self.source_encoder_decoders = nn.ModuleDict({
                st: EncoderDecoder(len(obs_source[st].columns), encoded_size = 64, num_step = 48, out_channels = 64, dropout = dropout) for st in self.source_stations
            })
            if bn:
                self.decoded_bn = nn.BatchNorm1d(64)

        self.wrf_cmaq_dlstms = nn.Sequential(
            BidirectionalLSTM(10, 64, bn),
            BidirectionalLSTM(64, 64, bn)
        )

        self.gcn = BipartiteLSTMGCN1D([64, 64, 64], len(source_loc), bn)

        self.finals = nn.Sequential(
            BidirectionalLSTM(128, 64, bn),
            Dense1D(in_features = 64, out_features = 2, num_step = 48, dropout = dropout, time_distributed = True, activation = None)
        )

    def forward(self, source_obs_dic: dict, target_wrf_cmaq_dic: dict, dist: torch.tensor):
        source_decoded = torch.stack([self.source_encoder_decoders[st](t) for st, t in source_obs_dic.items()], axis = 0)
        U, N, F, T = source_decoded.shape
        if hasattr(self, 'decoded_bn'):
            source_decoded = self.decoded_bn(source_decoded.view(U*N, F, T)).view(U, N, F, T)
        target_wrf_cmaq = torch.stack(list(target_wrf_cmaq_dic.values()), axis = 0)
        V, N, C, T = target_wrf_cmaq.shape
        assert dist.shape == (U, V)
        
        out_obs = self.gcn(source_decoded, dist) # V, N, -1, T
        out_wrf_cmaq = self.wrf_cmaq_dlstms(target_wrf_cmaq.view(V*N, C, T)).view(V, N, -1, T)
        out = self.finals(torch.cat([out_obs, out_wrf_cmaq], axis = 2).view(V*N, -1, T)).view(V, N, -1, T)
        return out  


if __name__ == '__main__':
    pass