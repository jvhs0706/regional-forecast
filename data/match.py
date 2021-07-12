from geopy.distance import geodesic
import pickle as pk
import numpy as np
from netCDF4 import Dataset 

def wll(filename = '/home/dataop/data/nmodel/wrf_fc/2014/201401/2014010212/wrfout_d03_2014-01-02_12:00:00'):
    ds = Dataset(filename)
    lat = ds['XLAT'][0]
    lon = ds['XLONG'][0]
    array = np.stack([lat, lon], axis = -1)    
    return array

def cll(filename = '/disk/hq246/hsunai/data/GRIDCRO2D.3km.20150115'):
    ds = Dataset(filename)
    lat = ds['LAT'][0,0]
    lon = ds['LON'][0,0]
    array = np.stack([lat, lon], axis = -1)
    return array

def grid_index(grid, target):
    m, n, _ = grid.shape
    dist_mat = np.zeros((m,n))
    distance = lambda z: geodesic(z, target).km
    dist_mat = np.apply_along_axis(distance, 2, grid)
    ind = dist_mat.argmin()
    return (ind//n, ind%n), geodesic(grid[ind//n, ind%n], target).km

wrf_latlon, cmaq_latlon = wll(), cll()
[lat_min, lon_min], [lat_max, lon_max] = np.maximum(wrf_latlon.min(axis = (0, 1)), cmaq_latlon.min(axis = (0, 1))),\
    np.minimum(wrf_latlon.max(axis = (0, 1)), cmaq_latlon.max(axis = (0, 1)))

if __name__ == '__main__':
    wrf_latlon, cmaq_latlon = wll(), cll()
    print(f'Latitude range: {(lat_min + lat_max)/2:.7g} +- {(lat_max - lat_min)/2:.7g}')
    print(f'Longitude range: {(lon_min + lon_max)/2:.7g} +- {(lon_max - lon_min)/2:.7g}')