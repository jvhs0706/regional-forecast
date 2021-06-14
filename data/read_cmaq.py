from datetime import date, timedelta
import time
from netCDF4 import Dataset
import numpy as np
import pickle as pk
import pandas as pd

begin_date = date(2015,1,1)
end_date = date(2020,12,31)

def _get_mid_dir (d):
    x=str(d.year) + '/'
    x+=str(d.year) + str(d.month).zfill(2) + '/'
    x+=str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2)
    return x

def _get_year_day (d):
    return str(d.year) + str((d-date(d.year - 1, 12, 31)).days).zfill(3)

def _read_dataset (folder_date, file_date, resolution):
    fnp = 'CCTM_V5g_ebi_cb05cl_ae5_aq_mpich2.ACONC.'
    dirprefix='/home/dataop/data/nmodel/cmaq_fc/'
    dirsuffix='12/'+str(resolution)+'km/'
    filename = dirprefix + _get_mid_dir(folder_date) + dirsuffix + fnp + _get_year_day(file_date)
    ds = Dataset(filename)
    return ds

def read_cmaq(species, begin_date, end_date, begin_hour, predict_length, resolution, match_dict, summation = False, verbose = True):
    dic = {st: [] for st in match_dict} if summation else {st: {sp: [] for sp in species} for st in match_dict} 
    _date = begin_date
    while _date <= end_date:
        if verbose and _date.day == 1:
            print(_date)
        folder_date = _date - timedelta(days = 2)
        
        temp_dic = {sp: [] for sp in species}
        for i in range(1, predict_length + 2):
            file_date = folder_date + timedelta(days = i)
            ds = _read_dataset(folder_date, file_date, resolution)
            for sp in species:
                temp_dic[sp].append(ds[sp][:].data)
            ds.close()
        
        for sp in species:
            temp_dic[sp] = np.concatenate(temp_dic[sp], axis = 0)[12+(begin_hour-8):12+(begin_hour-8)+24*predict_length,0,:,:]
        
        for st, (c0, c1) in match_dict.items():
            if summation:
                dic[st].append(np.stack([temp_dic[sp][:,c0,c1] for sp in species]).sum(axis = 0))
            else:
                for sp in species:
                    dic[st][sp].append(temp_dic[sp][:,c0,c1])
        
        _date += timedelta(days = 1)

    correct_shape = ((end_date-begin_date).days+1, 24*predict_length)
    
    for st in dic:
        if summation:
            dic[st] = np.stack(dic[st])
            assert dic[st].shape == correct_shape
        else:
            for sp in species:
                dic[st][sp] = np.stack(dic[st][sp])
                assert dic[st][sp].shape == correct_shape
    return dic

if __name__ == '__main__':
    with open('match_target.pkl', 'rb') as f:
        match_dict = pk.load(f)
        match_dict = {st: m[1] for st, m in match_dict.items()}
    
    cmaq_data = read_cmaq(['NO2', 'O3', 'SO2', 'CO'], begin_date, end_date, 9, 2, resolution = 3, match_dict=match_dict, summation=False)
    fspmc_cmaq_data = read_cmaq(['ASO4J', 'ASO4I', 'ANO3J', 'ANO3I', 'ANH4J', 'ANH4I', 'AXYL1J', 'AALKJ', 'AXYL2J', 'AXYL3J', 'ATOL1J', 'ATOL2J', 'ATOL3J', 'ABNZ1J', 'ABNZ2J', 'ABNZ3J', 'ATRP1J', 'ATRP2J', 'AISO1J', 'AISO2J', 'ASQTJ', 'AORGCJ', 'AORGPAJ', 'AORGPAI', 'AECJ', 'AECI', 'A25J', 'A25I', 'ANAJ', 'ANAI', 'ACLJ', 'AISO3J', 'AOLGAJ', 'AOLGBJ'],\
        begin_date, end_date, 9, 2, resolution=3, match_dict=match_dict, summation=True)
    for st in cmaq_data:
        cmaq_data[st]['FSPMC'] = fspmc_cmaq_data[st]
    with open('cmaq_data_target.pkl', 'wb') as f:
        pk.dump(cmaq_data, f)