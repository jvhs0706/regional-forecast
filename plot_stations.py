import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import argparse

def get_border(lat_min = -np.infty, lat_max = np.infty, lon_min= -np.infty, lon_max = np.infty):
    with open('./pearl_delta.txt', 'r') as f:
        txt = f.read().split('\n')
    _output, temp = [], {'lon':[], 'lat':[]}
    for t in txt:
        if len(t) == 0:
            _output += [(np.array(temp['lon']), np.array(temp['lat']))]
            temp = {'lon':[], 'lat':[]}
        else:
            _lon, _lat = t.split('     ')
            if lon_min <= float(_lon) <= lon_max and lat_min <= float(_lat) <= lat_max:
                temp['lon'].append(float(_lon))
                temp['lat'].append(float(_lat))
            else:
                _output += [(np.array(temp['lon']), np.array(temp['lat']))]
                temp = {'lon':[], 'lat':[]}

    return _output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    with open('./data/lat_lon_source.pkl', 'rb') as f:
        loc_dic = pk.load(f)
        plt.plot([loc[1] for loc in loc_dic.values()], [loc[0] for loc in loc_dic.values()], 'or', markersize = 2, label = 'Source stations')
        source_stations = list(loc_dic.keys())

    with open('./data/lat_lon_target.pkl', 'rb') as f:
        loc_dic = pk.load(f)
        plt.plot([loc[1] for st, loc in loc_dic.items() if st not in source_stations], [loc[0] for st, loc in loc_dic.items() if st not in source_stations], 'ob', markersize = 2, label = 'Non-source target stations')
    
    borders = get_border()
    for (blon, blat) in borders:
        plt.plot(blon, blat, 'k', linewidth=1)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Locations of the source stations and target stations')
    plt.legend()
    plt.savefig('./station_locations.png')