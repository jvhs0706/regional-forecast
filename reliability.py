from test import *

def regional_reliability(source_lat_lon, weight_decay, lat_range, lon_range):
    x, y = np.linspace(*lon_range, 100), np.linspace(*lat_range, 100)
    X, Y = np.meshgrid(x, y)
    out = np.zeros_like(X)
    for decay_factor, (lat, lon) in zip(weight_decay, source_lat_lon.values()):
        dist = np.apply_along_axis(lambda z: geodesic((z[1], z[0]), (lat, lon)).km, axis = 0, arr = np.stack([X, Y]))
        out += np.exp(-decay_factor * dist)
    return X, Y, out  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = 'Model name.')
    parser.add_argument('-bs', '--batch_size', type = int, help = 'Batch size for loading data.', default = 64)
    args = parser.parse_args()
    
    model = torch.load(f'./{args.model}_models/model.nctmo')
    model.eval()
    decay = model.gcn.distance_decay.detach().numpy()
    
    with open('./data/lat_lon_source.pkl', 'rb') as f:
        source_lat_lon = pk.load(f)
        source_stations = list(source_lat_lon.keys())
    
    with open(f'./{args.model}_models/train.txt', 'r') as f:
        train_stations = f.read().splitlines()
    with open(f'./{args.model}_models/test.txt', 'r') as f:
        test_stations = f.read().splitlines()
    
    with open('./data/lat_lon_target.pkl', 'rb') as f:
        target_lat_lon = pk.load(f) 
    
    with open(f'./{args.model}_models/source_normalizers.pkl', 'rb') as f:
        source_normalizers = pk.load(f)
    with open(f'./{args.model}_models/wrf_cmaq_normalizers.pkl', 'rb') as f:
        wrf_cmaq_normalizers = pk.load(f)
    
    with open(f'./{args.model}_models/temporal_split.pkl', 'rb') as f:
        split = pk.load(f)
        train_dates, test_dates = split['train'], split['test']
        print(train_dates.shape, test_dates.shape)

    target_stations = train_stations + test_stations
    dataset = RegionalDataset(target_stations)
    dataset.target_wrf_cmaq.normalizer = wrf_cmaq_normalizers
    for st in source_stations:
        dataset.source.station_datasets[st].normalizer = source_normalizers[st]
    
    subset = torch.utils.data.Subset(dataset, test_dates)
    
    dataloader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False)
    model_pred, ground_truth = {st: [] for st in target_stations}, {st: [] for st in target_stations}
    test_target_features = [dataset.target_wrf_cmaq.features.index('FSPMC'), dataset.target_wrf_cmaq.features.index('O3')]
    with torch.no_grad():
        for j, (X0, X1, obs) in enumerate(dataloader):
            pred = model(X0, X1, dataset.dist).detach().numpy()
            for k, st in enumerate(target_stations):
                model_pred[st].append(pred[k])
                ground_truth[st].append(obs[st].detach().numpy())
            print(f'Testing on batch {j}...')
    model_pred = {st: np.concatenate(arr) for st, arr in model_pred.items()}
    ground_truth = {st: np.concatenate(arr) for st, arr in ground_truth.items()}
    
    plt.contourf(*regional_reliability(source_lat_lon, decay, (lat_min, lat_max), (lon_min, lon_max)))
    plt.colorbar()
    plt.title('Reliability')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.plot([loc[1] for loc in source_lat_lon.values()], [loc[0] for loc in source_lat_lon.values()], 'or', markersize = 3, label = 'Source stations')
    borders = get_border()
    for (blon, blat) in borders:
        plt.plot(blon, blat, 'k', linewidth=1)
    plt.legend()
    plt.savefig(f'./{args.model}_models/reliability_map.png')
    plt.close()
    

    metric_values = {m.__name__: {st: m(model_pred[st].reshape(-1, 2, 2, 24), ground_truth[st].reshape(-1, 2, 2, 24), axis = (0, 3)) for st in target_stations} for m in metrics}
        
    reliability_values = np.exp(-decay[:, None] * dataset.dist.detach().numpy()).sum(axis = 0)
    reliability_values = {st: val for st, val in zip(target_stations, reliability_values)}
    
    fig, axes = plt.subplots(4, 4, figsize = (50, 50))
    for j, m in enumerate(metrics):
        for i, (sp, display_name, unit) in enumerate(zip(['fspmc', 'o3'], ['$\mathrm{PM}_{2.5}$', '$\mathrm{O}_3$'], ['$\mu g/m^3$', 'ppbv'])):
            axes[j, 2 * i + 1].sharey(axes[j, 2 * i])
            for t in range(2):
                axes[j, 2 * i + t].scatter([reliability_values[st] for st in train_stations if st in source_stations],
                    [metric_values[m.__name__][st][i, t] for st in train_stations if st in source_stations], s = 50, label = 'Source, training')
                axes[j, 2 * i + t].scatter([reliability_values[st] for st in train_stations if st not in source_stations], 
                    [metric_values[m.__name__][st][i, t] for st in train_stations if st not in source_stations], s = 50, label = 'Non-source, training')
                axes[j, 2 * i + t].scatter([reliability_values[st] for st in test_stations if st in source_stations], 
                    [metric_values[m.__name__][st][i, t] for st in test_stations if st in source_stations], s = 50, label = 'Source, held-out')
                axes[j, 2 * i + t].scatter([reliability_values[st] for st in test_stations if st not in source_stations], 
                    [metric_values[m.__name__][st][i, t] for st in test_stations if st not in source_stations], s = 50, label = 'Non-source, held-out')
                
                axes[j, 2 * i + t].legend(fontsize = 24)
                axes[j, 2 * i + t].set_xlabel('Reliability', fontsize = 28)
                axes[j, 2 * i + t].xaxis.set_tick_params(labelsize=24)
                axes[j, 2 * i + t].yaxis.set_tick_params(labelsize=24)
                u = f' ({unit})' if j < 2 else ' (%)' if j == 2 else ''

                axes[j, 2 * i + t].set_title(f'{display_name} {m.__name__}{u}, time-lags {24 * t} - {24 * t + 23} h', fontsize = 32)
    plt.tight_layout()
    plt.savefig(f'./{args.model}_models/metrics_reliability.png')
    plt.close()