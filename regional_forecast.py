from scipy.interpolate import griddata 
import datetime
from data.match import *
import argparse
from netCDF4 import Dataset
from dataset import RegionalDataset

import torch

from plot_stations import get_border
import matplotlib.pyplot as plt

import os

wrf_species = ['PSFC', 'U10', 'V10', 'T2', 'Q2']
cmaq_species = ['NO2', 'O3', 'SO2', 'CO', ('ASO4J', 'ASO4I', 'ANO3J', 'ANO3I', 'ANH4J', 'ANH4I', 'AXYL1J', 'AALKJ', 'AXYL2J', 'AXYL3J', 'ATOL1J', 'ATOL2J', 'ATOL3J', 'ABNZ1J', 'ABNZ2J', 'ABNZ3J', 'ATRP1J', 'ATRP2J', 'AISO1J', 'AISO2J', 'ASQTJ', 'AORGCJ', 'AORGPAJ', 'AORGPAI', 'AECJ', 'AECI', 'A25J', 'A25I', 'ANAJ', 'ANAI', 'ACLJ', 'AISO3J', 'AOLGAJ', 'AOLGBJ')]

lat_resolution, lon_resolution = 50, 80

lats, lons = np.meshgrid(np.linspace(lat_min, lat_max, lat_resolution), np.linspace(lon_min, lon_max, lon_resolution), indexing='ij')
cmaq_latlon = cll()



borders = get_border()

dataset_base_date = date(2015, 1, 4)
begin_date, end_date = date(2020, 1, 1), date(2020, 12, 31)

def get_wrf(date: datetime.date, begin_hour = 9, predict_length = 2):
    out = {sp: [] for sp in wrf_species}
    folder_date = date - datetime.timedelta(days=2)
    directory = f'/home/dataop/data/nmodel/wrf_fc/{str(folder_date.year)}/{str(folder_date.year) + str(folder_date.month).zfill(2)}/{str(folder_date.year) + str(folder_date.month).zfill(2) + str(folder_date.day).zfill(2)}12'
    for k in range(begin_hour-8, 24*predict_length+begin_hour-8):
        dt = datetime.datetime(date.year, date.month, date.day)+datetime.timedelta(hours=k)
        filename = f'wrfout_d03_{str(dt.year)}-{str(dt.month).zfill(2)}-{str(dt.day).zfill(2)}_{str(dt.time())}'
        fullpath = f'{directory}/{filename}'
        ds = Dataset(fullpath)
        wrf_lat, wrf_lon = ds['XLAT'][0], ds['XLONG'][0]
        for sp in wrf_species:
            pred = griddata((wrf_lat.flatten(), wrf_lon.flatten()), ds[sp][:].data[0].flatten(), (lats.flatten(), lons.flatten()), 'nearest').reshape(lat_resolution, lon_resolution)
            out[sp].append(pred)
        ds.close()
    
    for sp in out:
        out[sp] = np.stack(out[sp], axis = 0)
    return out

def get_cmaq(date: datetime.date, begin_hour = 9, predict_length = 2):
    temp_dic, out = {sp: [] for sp in cmaq_species if type(sp) == str}, {sp: [] for sp in cmaq_species if type(sp) == str}
    temp_dic['FSPMC'], out['FSPMC'] = [], []
    folder_date = date - datetime.timedelta(days=2)
    directory = f'/home/dataop/data/nmodel/cmaq_fc/{str(folder_date.year)}/{str(folder_date.year) + str(folder_date.month).zfill(2)}/{str(folder_date.year) + str(folder_date.month).zfill(2) + str(folder_date.day).zfill(2)}12/3km'
    
    for i in range(1, predict_length + 2):
        file_date = folder_date + datetime.timedelta(days = i)
        filename = f'CCTM_V5g_ebi_cb05cl_ae5_aq_mpich2.ACONC.{file_date.year}{str((file_date-datetime.date(file_date.year - 1, 12, 31)).days).zfill(3)}'
        fullpath = f'{directory}/{filename}'
        ds = Dataset(fullpath)
        for sp in cmaq_species:
            if type(sp) == str:
                temp_dic[sp].append(ds[sp][:].data)
            else:
                temp_dic['FSPMC'].append(np.sum(np.array([
                    ds[_sp][:].data for _sp in sp 
                ]), axis = 0))
        ds.close()
        
    for sp in temp_dic.keys():
        temp_arr = np.concatenate(temp_dic[sp], axis = 0)[12+(begin_hour-8):12+(begin_hour-8)+24*predict_length,0,:,:]
        for t in range(temp_arr.shape[0]):
            out[sp].append(griddata((cmaq_latlon[:, :, 0].flatten(), cmaq_latlon[:, :, 1].flatten()), temp_arr[t].flatten(), (lats.flatten(), lons.flatten()), 'nearest').reshape(lat_resolution, lon_resolution))
        out[sp] = np.stack(out[sp])
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = 'Model name.')
    parser.add_argument('-bs', '--batch_size', type = int, default = 2)
    args = parser.parse_args()
    
    with open(f'./{args.model}_models/train.txt', 'r') as f:
        train_stations = f.read().splitlines()
    with open(f'./{args.model}_models/test.txt', 'r') as f:
        test_stations = f.read().splitlines()
    with open(f'./{args.model}_models/source_normalizers.pkl', 'rb') as f:
        source_normalizers = pk.load(f)
    with open(f'./{args.model}_models/wrf_cmaq_normalizers.pkl', 'rb') as f:
        wrf_cmaq_normalizers = pk.load(f)
    with open('./data/lat_lon_source.pkl', 'rb') as f:
        source_lat_lon = pk.load(f)
        source_stations = list(source_lat_lon.keys())
    with open('./data/lat_lon_target.pkl', 'rb') as f:
        target_lat_lon = pk.load(f)
        target_stations = list(target_lat_lon.keys())
    with open(f'./{args.model}_models/baseline/baseline_source_differences.pkl', 'rb') as f:
        baseline_source_differences = pk.load(f)
    with open(f'./{args.model}_models/temporal_split.pkl', 'rb') as f:
        split = pk.load(f)
        train_dates, test_dates = split['train'], split['test']

    if not os.path.isdir(f'./{args.model}_models/regional_forecast_results'):
        os.makedirs(f'./{args.model}_models/regional_forecast_results')

    baseline_weights = np.stack([np.apply_along_axis(lambda z: 1/geodesic(z, source_lat_lon[s]).km, 2, np.stack([lats, lons], axis = -1)) for s in source_stations], axis = 0)
    baseline_weights = baseline_weights / baseline_weights.sum(axis = 0, keepdims = True)
    model = torch.load(f'./{args.model}_models/model.nctmo')
    
    dataset = RegionalDataset(target_stations)

    dist = torch.tensor([np.apply_along_axis(lambda z: geodesic(source_lat_lon[_s], z).km, -1, np.stack([lats, lons], axis = -1)).reshape(lat_resolution* lon_resolution) for _s in source_stations]).float()

    for st in source_stations:
        dataset.source.station_datasets[st].normalizer = source_normalizers[st]
    

    _date = begin_date
    while _date <= end_date:
        _ind = (_date - dataset_base_date).days
        X0, _, _y = dataset[_ind]
        for st in source_stations:
            X0[st] = X0[st].unsqueeze(0).float()
        
        X1 = {**get_wrf(_date), **get_cmaq(_date)}
        cmaq_out = np.stack([X1['FSPMC'].transpose(1, 2, 0), X1['O3'].transpose(1, 2, 0) * 1000], axis = -2)
        baseline_diff = np.stack([baseline_source_differences[s][_ind] for s in source_stations])
        diff = (baseline_weights[:, :, :, None, None] * baseline_diff[:, None, None, :, :]).sum(axis = 0)
        baseline_out = cmaq_out + diff

        X1 = torch.tensor(np.stack([X1[sp].reshape(-1, lat_resolution* lon_resolution).T for sp in dataset.target_wrf_cmaq.features], axis = -2)).unsqueeze(1).float()
        X1 = wrf_cmaq_normalizers(X1)

        

        source_decoded = torch.stack([model.source_encoder_decoders[st](t) for st, t in X0.items()], axis = 0)
        U, N, F, T = source_decoded.shape
        if hasattr(model, 'decoded_bn'):
            source_decoded = model.decoded_bn(source_decoded.view(U*N, F, T)).view(U, N, F, T)
        
        V, N, C, T = X1.shape
        assert dist.shape == (U, V)
        out_obs = model.gcn(source_decoded, dist) # V, N, -1, T
        out_wrf_cmaq = model.wrf_cmaq_dlstms(X1.view(V*N, C, T)).view(V, N, -1, T)
        out = model.finals(torch.cat([out_obs, out_wrf_cmaq], axis = 2).view(V*N, -1, T)).view(lat_resolution, lon_resolution, N, -1, T)
        out = out.squeeze(2).detach().numpy()

        np.save(f'./{args.model}_models/regional_forecast_results/NCTMO.CMAQ.{_date.year}{_date.month:02}{_date.day:02}09.npy', cmaq_out.astype(np.float16))
        np.save(f'./{args.model}_models/regional_forecast_results/NCTMO.BASELINE.{_date.year}{_date.month:02}{_date.day:02}09.npy', baseline_out.astype(np.float16))
        np.save(f'./{args.model}_models/regional_forecast_results/NCTMO.MODEL.{_date.year}{_date.month:02}{_date.day:02}09.npy', out.astype(np.float16))

        _date = _date + datetime.timedelta(days = 1)