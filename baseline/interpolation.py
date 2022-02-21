import numpy as np
import argparse
import pickle as pk
from geopy.distance import geodesic

from dataset import *

from pykrige.ok import OrdinaryKriging
from scipy.interpolate import SmoothSphereBivariateSpline

class Interpolation:
    def __init__(self, lat_lon, values):
        N, _ = lat_lon.shape
        self.lat_lon = lat_lon # (N, 2)
        self.values = values # (N, *)

    def _interpolate(self, target_lat_lon, z):
        raise NotImplementedError

    def interpolate(self, target_lat_lon):
        M, _ = target_lat_lon.shape # (M, 2)
        return np.apply_along_axis(lambda z: self._interpolate(target_lat_lon, z), 0, self.values)

class NearestNeighbor(Interpolation):
    def __init__(self, lat_lon, values):
        super().__init__(lat_lon, values)
        
    def _interpolate(self, target_lat_lon, z):
        indices = np.apply_along_axis(lambda t: np.apply_along_axis(lambda s: geodesic(s, t).km, 1, self.lat_lon).argmin(), 1, target_lat_lon)
        return z[indices]

class InverseDistanceWeighted(Interpolation):
    def __init__(self, lat_lon, values):
        super().__init__(lat_lon, values)
        
    def _interpolate(self, target_lat_lon, z):
        dist = np.apply_along_axis(lambda t: np.apply_along_axis(lambda s: geodesic(s, t).km, 1, self.lat_lon), 1, target_lat_lon) # (M, N)
        weights = (1/dist) / (1/dist).sum(axis = 1, keepdims = True)
        return weights @ z

class Kriging(Interpolation):
    def __init__(self, lat_lon, values):
        super().__init__(lat_lon, values)

    def _interpolate(self, target_lat_lon, z):
        kriging_instance = OrdinaryKriging(x = self.lat_lon[:, 1], y = self.lat_lon[:, 0], z = z, variogram_model='spherical', coordinates_type = 'geographic')
        out, _ = kriging_instance.execute('points', target_lat_lon[:, 1], target_lat_lon[:, 0])
        return out

interpolation_methods = {
    'NN': NearestNeighbor,
    'IDW': InverseDistanceWeighted,
    'Kriging': Kriging
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = 'Model name.')
    parser.add_argument('method', help = 'Interpolation method.')
    args = parser.parse_args()

    with open('../data/lat_lon_source.pkl', 'rb') as f:
        source_lat_lon = pk.load(f)
        source_stations = list(source_lat_lon.keys())   

    with open(f'../{args.model}_models/train.txt', 'r') as f:
        train_stations = f.read().splitlines()
    with open(f'../{args.model}_models/test.txt', 'r') as f:
        test_stations = f.read().splitlines()
    target_stations = [st for st in test_stations if st not in source_stations]
    print(f'{len(target_stations)} to be tested...')

    with open(f'../{args.model}_models/temporal_split.pkl', 'rb') as f:
        split = pk.load(f)
        train_dates, test_dates = split['train'], split['test']

    with open('../data/lat_lon_target.pkl', 'rb') as f:
        target_lat_lon = pk.load(f) 

    with open(f'../{args.model}_models/baseline/baseline_source_predictions.pkl', 'rb') as f:
        baseline_source_predictions = pk.load(f)
        diff_arr = np.stack([baseline_source_predictions[st][0][test_dates]-baseline_source_predictions[st][1][test_dates] for st in source_stations])

    with open('../data/cmaq_data_target.pkl', 'rb') as f:
        cmaq_dic = pk.load(f)
        cmaq_arr = []
        for st in target_stations:
            cmaq_arr.append([cmaq_dic[st][sp][test_dates + history] for sp in target_species_cmaq])
    cmaq_arr = np.array(cmaq_arr).transpose(0, 2, 1, 3) * np.array([1, 1000])[None, None, :, None]

    interp = interpolation_methods[args.method](np.stack(source_lat_lon.values()), np.stack([baseline_source_predictions[st][0][test_dates]-baseline_source_predictions[st][1][test_dates] for st in source_stations]))
    output_arr = interp.interpolate(np.stack([target_lat_lon[st] for st in target_stations]))
    output_arr = output_arr + cmaq_arr
    assert len(output_arr) == len(target_stations)
    output_dic = {}
    for st, arr in zip(target_stations, output_arr):
        output_dic[st] = arr
    with open(f'../{args.model}_models/baseline/baseline_{args.method}_predictions.pkl', 'wb') as f:
        pk.dump(output_dic, f)