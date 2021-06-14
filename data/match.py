from geopy.distance import geodesic
import pickle as pk
import numpy as np
from netCDF4 import Dataset 

def wll(filename):
    ds = Dataset(filename)
    lat = ds['XLAT'][0]
    lon = ds['XLONG'][0]
    array = np.stack([lat, lon], axis = -1)    
    return array

def cll(filename):
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
