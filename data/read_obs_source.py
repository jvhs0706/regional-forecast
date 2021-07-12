import os
import pandas as pd 
import numpy as np 
import pickle as pk
from datetime import datetime
import argparse
from geopy.distance import geodesic
from match import *

target_species = ['AQ_FSPMC', 'AQ_O3']
other_species = ['AQ_NO2', 'AQ_SO2', 'AQ_CO', 'A_TEMP', 'A_WIND', 'A_PRECIP', 'A_DEW_PT', 'A_PRE_SLP', 'A_RH_VT']

data_dir = '../../data'
dataset_ndays = 2192

def remove_invalid(x):
    try:
        return float(x) if x != -99999.0 else np.NaN
    except:
        return np.NaN

def read_csv(fn: str, index_row: int, ndays: int, threshold: float, interpolate: int, begin_dt: datetime = None, end_dt: datetime = None):
    loc = pd.read_csv(fn, skiprows = lambda x: x not in range(index_row, index_row + 3), index_col = 0).applymap(remove_invalid)
    df = pd.read_csv(fn, skiprows = lambda x: x != index_row and x not in range(index_row + 4, index_row + 4 + 24 * ndays), index_col = 0).applymap(remove_invalid)
    df['time'] = pd.to_datetime(df.index.values, format = '%Y/%m/%d %H:%M:%S')
    df.set_index('time', inplace = True)
    df = df.loc[begin_dt:end_dt]
    if interpolate > 0:
        df_interpolated = df.interpolate(limit_direction='both', limit = interpolate)
        mask = np.isnan(df_interpolated.values).sum(axis = 0) / df.shape[0] <= threshold
    else:
        mask = np.isnan(df.values).sum(axis = 0) / df.shape[0] <= threshold
    loc, df = loc.iloc[:, mask], df.iloc[:, mask]
    return loc, df

def merge_category(species: list, loc_dic: dict, data_dic: dict):
    stations = list(set.intersection(*[set(loc.columns) for loc in loc_dic.values()]))
    loc, data = next(iter(loc_dic.values())), {}
    
    loc = {st: tuple(loc[st]) for st in stations}
    index = next(iter(data_dic.values())).index
    
    for st in stations:
        df = pd.DataFrame.from_dict({
            sp: data_dic[sp][st] for sp in species            
        }).set_index(index)
        data[st] = df

    return stations, loc, data

def _read_target (threshold: float):
    loc_dic, data_dic = {}, {}
    for sp in target_species:
        loc_dic[sp], data_dic[sp] = read_csv(fn= f'{data_dir}/{sp}-20150101-20201231.csv',
            index_row=5, ndays = dataset_ndays, threshold=threshold, interpolate=0)

    target_stations, target_loc, target_data = merge_category(species=target_species, loc_dic=loc_dic, data_dic=data_dic)
    return target_stations, target_loc, target_data

def _read_other (threshold: float, interpolate: int):
    loc_dic, data_dic = {}, {}
    for sp in other_species:
        _interp = 0 if sp[:2] == 'AQ' else interpolate
        if sp != 'A_WIND':
            loc_dic[sp], data_dic[sp] = read_csv(fn= f'{data_dir}/{sp}-20150101-20201231.csv', index_row=5, ndays = dataset_ndays, threshold=threshold, interpolate=_interp)
        else:
            loc_dic[sp], data_dic['A_WIND_S'] = read_csv(fn= f'{data_dir}/{sp}-20150101-20201231.csv', index_row=5, ndays = dataset_ndays, threshold=threshold, interpolate=_interp)
            loc_dic[sp], data_dic['A_WIND_D'] = read_csv(fn= f'{data_dir}/{sp}-20150101-20201231.csv', index_row=52619, ndays = dataset_ndays, threshold=threshold, interpolate=_interp)

    return loc_dic, data_dic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interpolate', type = int, help = 'Interpolate the meterological factor observations due to different temporal resolutions.', default = 2)
    parser.add_argument('--invalid_threshold', type = float, help = 'Maximum ratio of invalid values allowed for each entry of observation.', default = 0.1)
    parser.add_argument('--match_distance', type = float, help = 'Maximum distance allowed for matching.', default=10.0)
    parser.add_argument('--feature_threshold', type = int, help = 'The minimum number of features.', default=8)
    args = parser.parse_args()
    
    target_stations, target_loc, target_data = _read_target(threshold=args.invalid_threshold)
    other_loc_dic, other_data_dic = _read_other (threshold=args.invalid_threshold, interpolate=args.interpolate)
    wrf_latlon, cmaq_latlon = wll(), cll()

    valid_loc, valid_data, match = {}, {}, {}
    for st in target_stations:
        print(f'Processing station {st}...')
        stations_matched = {}
        df_temp = target_data[st].copy()
        for sp in other_species:
            dist = lambda s: geodesic(other_loc_dic[sp][s].values, target_loc[st]).km
            matched = min(list(other_loc_dic[sp].columns), key = dist)
            
            if dist(matched) < args.match_distance:
                if sp != 'A_WIND':
                    df_temp[sp] = other_data_dic[sp][matched]
                else:
                    df_temp['A_WIND_S'], df_temp['A_WIND_D'] = other_data_dic['A_WIND_S'][matched], other_data_dic['A_WIND_D'][matched]
                stations_matched[sp] = matched
        
        wind, wdis = grid_index(wrf_latlon, target_loc[st])
        cind, cdis = grid_index(cmaq_latlon, target_loc[st])      
        
        if len(df_temp.columns) >= args.feature_threshold and max([wdis, cdis]) < args.match_distance:
            valid_loc[st] = target_loc[st]
            valid_data[st] = df_temp
            match[st] = (stations_matched, wind, cind)

    with open('./lat_lon_source.pkl', 'wb') as f:
        pk.dump(valid_loc, f)
    with open('./obs_data_source.pkl', 'wb') as f:
        pk.dump(valid_data, f)
    with open('./match_source.pkl', 'wb') as f:
        pk.dump(match, f)

    print(f'Total number of source stations: {len(match)}.')