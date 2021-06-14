import os
import pandas as pd 
import numpy as np 
import pickle as pk
from datetime import datetime
import argparse
from geopy.distance import geodesic

from read_obs_source import read_csv, merge_category, data_dir, dataset_ndays
from match import *

target_species = ['AQ_FSPMC', 'AQ_O3']

def _read_target (threshold: float):
    loc_dic, data_dic = {}, {}
    for sp in target_species:
        loc_dic[sp], data_dic[sp] = read_csv(fn= f'{data_dir}/{sp}-20150101-20201231.csv',
            index_row=5, ndays = dataset_ndays, threshold=threshold, interpolate=0)

    target_stations, target_loc, target_data = merge_category(species=target_species, loc_dic=loc_dic, data_dic=data_dic)
    return target_stations, target_loc, target_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--invalid_threshold', type = float, help = 'Maximum ratio of invalid values allowed for each entry of observation.', default = 0.3)
    parser.add_argument('--match_distance', type = float, help = 'Maximum distance allowed for matching.', default=10.0)
    args = parser.parse_args()
    
    target_stations, target_loc, target_data = _read_target(threshold=args.invalid_threshold)
    wrf_latlon, cmaq_latlon = wll('/home/dataop/data/nmodel/wrf_fc/2014/201401/2014010212/wrfout_d03_2014-01-02_12:00:00'), cll('GRIDCRO2D.3km.20150115')

    valid_loc, valid_data, match = {}, {}, {}
    for st in target_stations:
        print(f'Processing station {st}...')
        wind, wdis = grid_index(wrf_latlon, target_loc[st])
        cind, cdis = grid_index(cmaq_latlon, target_loc[st])      
        
        if max([wdis, cdis]) < args.match_distance:
            valid_loc[st] = target_loc[st]
            valid_data[st] = target_data[st]
            match[st] = (wind, cind)

    with open('./lat_lon_target.pkl', 'wb') as f:
        pk.dump(valid_loc, f)
    with open('./obs_data_target.pkl', 'wb') as f:
        pk.dump(valid_data, f)
    with open('./match_target.pkl', 'wb') as f:
        pk.dump(match, f)

    print(f'Total number of stations: {len(match)}.')