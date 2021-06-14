import torch
import torch.nn as nn
import torch.nn.functional as F

class UDownBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bn = True):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding = (kernel_size-1)//2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        if bn:
            self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        cout = F.relu(self.conv1d(x))
        pout = self.maxpool1d(cout)
        if hasattr(self, 'bn'):
            pout = self.bn(pout)
        return cout, pout

class ULevelBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bn = True):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding = (kernel_size-1)//2)
        if bn:
            self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        out = F.relu(self.conv1d(x))
        if hasattr(self, 'bn'):
            out = self.bn(out)
        return out 

class UUpBlock1D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size: int = 3, bn = True):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=kernel_size, padding = (kernel_size-1)//2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'linear', align_corners = True)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding = (kernel_size-1)//2)
        if bn:
            self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x1, x2):
        N1, C1, T1 = x1.shape 
        N2, C2, T2 = x2.shape 
        assert C1 == 2*C2 and T2 == 2*T1
        out1 = self.upsample(F.relu(self.conv1(x1)))
        out = torch.cat([out1, x2], dim = 1)
        out = F.relu(self.conv2(out))
        if hasattr(self, 'bn'):
            out = self.bn(out)
        return out

def _module_list (module: type, in_channels: list, out_channels:list):
    assert(len(in_channels) == len(out_channels))
    return nn.ModuleList([module(_in, _out) for _in, _out in zip(in_channels, out_channels)])

class UNet1D(nn.Module):
    def __init__ (self, in_channels: list, down_channels: list, level_channels: list, up_channels:list):
        super().__init__()
        self.downblocks = _module_list(UDownBlock1D, [in_channels]+down_channels[:-1], down_channels)
        self.bottomblocks = _module_list(ULevelBlock1D, [down_channels[-1]]+level_channels[:-1], level_channels)
        self.upblocks = _module_list(UUpBlock1D, [level_channels[-1]]+up_channels[:-1], up_channels)

    def forward(self, x):
        skips = []
        out = x
        for m in self.downblocks:
            pout, out = m(out)
            skips.append(pout)
        for m in self.bottomblocks:
            out = m(out)
        for m, pout in zip(self.upblocks, reversed(skips)):
            out = m(out, pout)
        return out