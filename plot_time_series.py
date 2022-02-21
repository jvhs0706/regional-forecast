from test import *

date_zero = date(2020, 1, 2)
date_last = date_zero + timedelta(days = 364)

time_axis = np.array([date_zero + timedelta(days = h) for h in range(364)]) 

def plot_ax(ax, title, time_axis, begin_date = date_zero, end_date = date_last, *, obs, pred, cmaq, bl):
    begin_index, end_index = (begin_date - date_zero).days, (end_date - date_zero).days + 1

    ax.plot(time_axis[begin_index:end_index], cmaq[begin_index:end_index], 'y', label = 'CMAQ')
    ax.plot(time_axis[begin_index:end_index], bl[begin_index:end_index], 'c', label = 'Spatial correction')
    ax.plot(time_axis[begin_index:end_index], pred[begin_index:end_index], 'b', label = 'Broadcasting')
    ax.plot(time_axis[begin_index:end_index], obs[begin_index:end_index], 'ro', label = 'Observation', markersize = 2)
    
    ax.legend(fontsize = 20)
    ax.set_title(title, fontsize = 24)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = 'Model name.')
    parser.add_argument('-bs', '--batch_size', type = int, help = 'Batch size for loading data.', default = 64)
    parser.add_argument('-bl', '--baseline', type = str, help = 'Type of prediction.', default = 'Kriging')
    parser.add_argument('--begin_date', type = int, nargs = 3, help = 'Day 0 of plot.', default = [date_zero.year, date_zero.month, date_zero.day])
    parser.add_argument('--end_date', type = int, nargs = 3, help = 'Day -1 of plot.', default = [date_last.year, date_last.month, date_last.day])
    parser.add_argument('--stations', type = str, nargs = '*', help = 'Plot certain stations.')
    args = parser.parse_args()

    begin_date, end_date = date(*args.begin_date), date(*args.end_date)

    # model loading
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
        test_stations = [st for st in test_stations if st not in source_stations]     
    
    with open('./data/lat_lon_target.pkl', 'rb') as f:
        target_lat_lon = pk.load(f) 
    
    with open(f'./{args.model}_models/source_normalizers.pkl', 'rb') as f:
        source_normalizers = pk.load(f)
    with open(f'./{args.model}_models/wrf_cmaq_normalizers.pkl', 'rb') as f:
        wrf_cmaq_normalizers = pk.load(f)
    
    with open(f'./{args.model}_models/temporal_split.pkl', 'rb') as f:
        split = pk.load(f)
        train_dates, test_dates = split['train'], split['test']

    with open(f'./{args.model}_models/baseline/baseline_{args.baseline}_predictions.pkl', 'rb') as f:
        baseline_pred = pk.load(f)


    test = RegionalDataset(test_stations)
    test_subset = torch.utils.data.Subset(test, test_dates)
    test.target_wrf_cmaq.normalizer = wrf_cmaq_normalizers
    for st in source_stations:
        test.source.station_datasets[st].normalizer = source_normalizers[st]
    
    test_dataloader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
    test_pred, cmaq_pred, ground_truth = {st: [] for st in test_stations}, {}, {st: [] for st in test_stations}
    test_target_features = [test.target_wrf_cmaq.features.index('FSPMC'), test.target_wrf_cmaq.features.index('O3')]
    with torch.no_grad():
        for j, (X0, X1, obs) in enumerate(test_dataloader):
            pred = model(X0, X1, test.dist).detach()
            for k, st in enumerate(test_stations):
                test_pred[st].append(pred[k])
                ground_truth[st].append(obs[st].detach())
            print(f'Testing on test set, batch {j}...')
    test_pred = {st: np.concatenate(arr) for st, arr in test_pred.items()}
    ground_truth = {st: np.concatenate(arr) for st, arr in ground_truth.items()}

    for st in test_stations:
        ds = test.target_wrf_cmaq.station_datasets[st]
        cmaq_pred[st] = np.stack(
                [
                    ds.wrf_cmaq[ds.history: ds.history + len(ds), test_target_features[0], :],
                    1000 * ds.wrf_cmaq[ds.history: ds.history + len(ds), test_target_features[1], :]
                ], axis = 1
            )[test_dates]

    if not os.path.isdir(f'./{args.model}_models/plots'):
        os.makedirs(f'./{args.model}_models/plots')
    if args.stations is None:
        for st in test_stations:
            lat, lon = target_lat_lon[st]
            fig, axes = plt.subplots(2, 2, figsize = (50, 20))
            fig.suptitle(f'Station {st} ({lat:.2f}°N, {lon:.2f}°E), {str(begin_date)} - {str(end_date)}', fontsize = 36)
            for i, (sp, display_name, unit) in enumerate(zip(['FSPMC', 'O3'], ['$\mathrm{PM}_{2.5}$', '$\mathrm{O}_3$'], ['$\mu g/m^3$', 'ppbv'])):
                obs = np.nanmean(ground_truth[st][1:, i, :24], axis = -1)
                pred = [test_pred[st][1:, i, :24].mean(axis = -1), test_pred[st][:-1, i, 24:].mean(axis = -1)]
                cmaq = [cmaq_pred[st][1:, i, :24].mean(axis = -1), cmaq_pred[st][:-1, i, 24:].mean(axis = -1)]
                bl = [baseline_pred[st][1:, i, :24].mean(axis = -1), baseline_pred[st][:-1, i, 24:].mean(axis = -1)]
                axes[i, 1].sharey(axes[i, 0])
                for j in range(2):
                    plot_ax(axes[i, j], f'{display_name} ({unit}), time-lags {24 * j} - {24 * j + 23} h', time_axis, begin_date, end_date, obs = obs, pred = pred[j], cmaq = cmaq[j], bl = np.maximum(bl[j], 0))
            plt.tight_layout()
            plt.savefig(f'./{args.model}_models/plots/{st}_{str(begin_date)}_{str(end_date)}.png')
            plt.close()
    else:
        for i, (sp, display_name, unit) in enumerate(zip(['FSPMC', 'O3'], ['$\mathrm{PM}_{2.5}$', '$\mathrm{O}_3$'], ['$\mu g/m^3$', 'ppbv'])):
            fig = plt.Figure(figsize = (20 * len(args.stations), 30), constrained_layout= True)
            cols = fig.subfigures(1, len(args.stations))
            for j, st in enumerate(args.stations):
                lat, lon = target_lat_lon[st]
                cols[j].suptitle(f'Station {st} ({lat:.2f}°N, {lon:.2f}°E)', fontsize = 30)
                axes = cols[j].subplots(2, 1)
                obs = np.nanmean(ground_truth[st][1:, i, :24], axis = -1)
                pred = [test_pred[st][1:, i, :24].mean(axis = -1), test_pred[st][:-1, i, 24:].mean(axis = -1)]
                cmaq = [cmaq_pred[st][1:, i, :24].mean(axis = -1), cmaq_pred[st][:-1, i, 24:].mean(axis = -1)]
                bl = [baseline_pred[st][1:, i, :24].mean(axis = -1), baseline_pred[st][:-1, i, 24:].mean(axis = -1)]
                for k in range(2):
                    plot_ax(axes[k], f'Time-lags {24 * k} - {24 * k + 23} h', time_axis, begin_date, end_date, obs = obs, pred = pred[k], cmaq = cmaq[k], bl = np.maximum(bl[k], 0))
            fig.savefig(f'./{args.model}_models/plots/{sp}_{str(begin_date)}_{str(end_date)}.png')
            plt.close(fig)