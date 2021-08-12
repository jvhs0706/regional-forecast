from datetime import date, datetime, timedelta
from netCDF4 import Dataset
import numpy as np
import pickle as pk
from pandas import DataFrame

dirprefix = '/home/dataop/data/nmodel/wrf_fc/'
dirsuffix = '12/'

begin_date = date(2015,1,1)
end_date = date(2020,12,31)

def _get_mid_dir (d):
    x=str(d.year) + '/'
    x+=str(d.year) + str(d.month).zfill(2) + '/'
    x+=str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2)
    return x

def _get_filename (dt, pred_class):
    return 'wrfout_'+pred_class+'_'+str(dt.year).zfill(2)+'-'+str(dt.month).zfill(2)+'-'+str(dt.day).zfill(2)+'_'+str(dt.time())

def read_wrf(species, begin_date, end_date, begin_hour, predict_length, pred_class, match_dict, verbose = True):
    dic = {st: {sp: [] for sp in species} for st in match_dict}
    _date = begin_date
    while _date <= end_date:
        if verbose and _date.day == 1:
            print(_date)
        filedir = dirprefix + _get_mid_dir(_date-timedelta(days=2)) + dirsuffix
        temp_dic = {sp: [] for sp in species}
        for k in range(begin_hour-8, 24*predict_length+begin_hour-8):
            fullpath = filedir+_get_filename(datetime(_date.year, _date.month, _date.day)+timedelta(hours=k), pred_class)
            ds = Dataset(fullpath)
            for sp in species:
                temp_dic[sp].append(ds[sp][:].data)
            ds.close()

        for sp in species:
            temp_dic[sp] = np.concatenate(temp_dic[sp], axis = 0)
            
        for st, (c0, c1) in match_dict.items():
            for sp in species:    
                dic[st][sp].append(temp_dic[sp][:, c0, c1])
        
        _date += timedelta(days=1)
    
    for st in dic:
        for sp in species:
            dic[st][sp] = np.stack(dic[st][sp])
            assert(dic[st][sp].shape == ((end_date-begin_date).days+1, 24*predict_length))
    return dic

if __name__ == '__main__':
    with open('match_target.pkl', 'rb') as f:
        match_dict = pk.load(f)
        match_dict = {st: m[0] for st, m in match_dict.items()}
    
    wrf_data = read_wrf(['PSFC', 'U10', 'V10', 'T2', 'Q2'], begin_date, end_date, 9, 2, 'd03', match_dict)
    with open('wrf_data_target.pkl', 'wb') as f:
        pk.dump(wrf_data, f)

    