from scipy.interpolate import griddata 
import datetime
from data.match import *
import argparse
from netCDF4 import Dataset
from dataset import RegionalDataset

import torch

from plot_stations import get_border
import matplotlib.pyplot as plt
import itertools
from regional_forecast import lats, lons

pred_classes = ['CMAQ', 'BASELINE', 'MODEL']

fspmc_cm_kwargs = {'vmin': 0, 'vmax': 60, 'cmap': 'GnBu'}
o3_cm_kwargs = {'vmin': 0, 'vmax': 50, 'cmap': 'GnBu'}

borders = get_border()

def _get_forecast(date, pred_class):
    _date0, _date1 = date, date - datetime.timedelta(days = 1)
    fn0 = f'./{args.model}_models/regional_forecast_results/NCTMO.{pred_class}.{_date0.year}{_date0.month:02}{_date0.day:02}09.npy'
    fn1 = f'./{args.model}_models/regional_forecast_results/NCTMO.{pred_class}.{_date1.year}{_date1.month:02}{_date1.day:02}09.npy'
    arr0 = np.load(fn0)
    arr1 = np.load(fn1)
    return {
        'fspmc.day0': arr0[:, :, 0, :24],
        'fspmc.day1': arr1[:, :, 0, 24:],
        'o3.day0': arr0[:, :, 1, :24],
        'o3.day1': arr1[:, :, 1, 24:]
    }

def get_forecast(date):
    out = {}
    for pred_class in pred_classes:
        class_dic = _get_forecast(date, pred_class)
        for k, arr in class_dic.items():
            out[f'{pred_class}.{k}'] = arr 
    return out

def get_ground_observation(begin_date, end_date):
    begin_index = (begin_date - datetime.date(2015, 1, 1)).days * 24 
    end_index = ((end_date - datetime.date(2015, 1, 1)).days + 1) * 24
    with open('./data/lat_lon_target.pkl', 'rb') as f:
        target_lat_lon = pk.load(f)
        target_stations = list(target_lat_lon.keys())
    
    target_out_dic = {}
    with open('./data/obs_data_target.pkl', 'rb') as f:
        obs_data_target = pk.load(f)
        for st in target_stations:
            arr = np.nanmean(obs_data_target[st].values[begin_index:end_index], axis = 0)
            print(st, arr)
            target_out_dic[st] = arr

    return target_stations, target_lat_lon, target_out_dic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = 'Model name.')
    parser.add_argument('--begin_date', type = int, nargs = 3)
    parser.add_argument('--end_date', type = int, nargs = 3)
    args = parser.parse_args()

    begin_date, end_date = datetime.date(*args.begin_date), datetime.date(*args.end_date)

    N_dates = (end_date - begin_date).days + 1

    out_dict = {f'{m}.{p}.{d}':np.zeros_like(lats) for m, p, d in itertools.product(pred_classes, ['fspmc', 'o3'], ['day0', 'day1'])}
    target_stations, target_lat_lon, target_out_dic = get_ground_observation(begin_date, end_date)

    _date = begin_date
    while _date <= end_date:
        pred = get_forecast(_date)
        for k, arr in pred.items():
            out_dict[k] += arr.mean(axis = -1) / N_dates
        _date += datetime.timedelta(days = 1)

    for k, arr in out_dict.items():
        print(k, arr.mean())

    fig, axes = plt.subplots(4, 3, figsize = (100, 100))
    display_name = {'fspmc': '$\mathrm{PM}_{2.5}$', 'o3': '$\mathrm{O}_3$'}
    units = {'fspmc': '$\mu g/m^3$', 'o3': 'ppbv'}

    for i, (pol, d) in enumerate(itertools.product(['fspmc', 'o3'], ['day0', 'day1'])):
        cm_kwargs = fspmc_cm_kwargs if pol == 'fspmc' else o3_cm_kwargs

        for j, pred_class in enumerate(pred_classes):
            cm = axes[i, j].pcolormesh(lons, lats, out_dict[f'{pred_class}.{pol}.{d}'], **cm_kwargs)
        
        axes[i, 0].set_title(f'{display_name[pol]}, {24*(i % 2)} - {24*(i % 2) + 23} h, CMAQ', fontsize = 60)
        axes[i, 1].set_title(f'{display_name[pol]}, {24*(i % 2)} - {24*(i % 2) + 23} h, Spatial correction', fontsize = 60)
        axes[i, 2].set_title(f'{display_name[pol]}, {24*(i % 2)} - {24*(i % 2) + 23} h, Model', fontsize = 60)
        
        cb = fig.colorbar(cm, ax = axes[i, :], orientation = 'vertical')
        cb.ax.tick_params(labelsize = 44)
        cb.set_label(units[pol], fontsize = 48)

        for j in range(3):
            axes[i, j].scatter(x = [target_lat_lon[st][1] for st in target_stations], y = [target_lat_lon[st][0] for st in target_stations], \
                c = [target_out_dic[st][i // 2] for st in target_stations], s = 144, edgecolors='k', **cm_kwargs)
            for (blon, blat) in borders:
                axes[i, j].plot(blon, blat, 'k', linewidth=1)
            
            axes[i, j].xaxis.set_tick_params(labelsize=44)
            axes[i, j].yaxis.set_tick_params(labelsize=44)
            axes[i, j].set_xlabel('Longitude', fontsize=48)
            axes[i, j].set_ylabel('Latitude', fontsize=48)

    plt.savefig(f'./{args.model}_models/regional_forecast_{str(begin_date)}-{str(end_date)}.png', bbox_inches='tight')
