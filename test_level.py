from test import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = 'Model name.')
    parser.add_argument('-bs', '--batch_size', type = int, help = 'Batch size for loading data.', default = 64)
    parser.add_argument('-bl', '--baselines', nargs = '*', help = 'Load baseline data.', default = [])
    parser.add_argument('-fspmc', '--fspmc_levels', type = float, nargs = '*', help = 'The segments of FSPMC levels.', default = [])
    parser.add_argument('-o3', '--o3_levels', type = float, nargs = '*', help = 'The segments of O3 levels.', default = [])

    args = parser.parse_args()
    
    # model loading
    model = torch.load(f'./{args.model}_models/model.nctmo')
    model.eval()
    
    with open('./data/lat_lon_source.pkl', 'rb') as f:
        source_lat_lon = pk.load(f)
        source_stations = list(source_lat_lon.keys())
    
    with open(f'./{args.model}_models/train.txt', 'r') as f:
        train_stations = f.read().splitlines()
    with open(f'./{args.model}_models/test.txt', 'r') as f:
        test_stations = f.read().splitlines()
        if len(args.baselines) > 0:
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

    test = RegionalDataset(test_stations)
    test_subset = torch.utils.data.Subset(test, test_dates)
    test.target_wrf_cmaq.normalizer = wrf_cmaq_normalizers
    for st in source_stations:
        test.source.station_datasets[st].normalizer = source_normalizers[st]
    
    # load the test predictions, cmaq predicions, and the ground truth
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

    # If needed, load the baseline predictions
    baseline_pred = {}
    for bl in args.baselines:
        with open(f'./{args.model}_models/baseline/baseline_{bl}_predictions.pkl', 'rb') as f:
            baseline_pred[bl] = pk.load(f)

    pred, y, cmaq = np.concatenate([test_pred[st] for st in test_stations], axis = 0),\
        np.concatenate([ground_truth[st] for st in test_stations], axis = 0), np.concatenate([cmaq_pred[st] for st in test_stations], axis = 0)
    assert y.shape == pred.shape == cmaq.shape
    N, _, _ = y.shape
    y, pred, cmaq = y.reshape(N, 2, 2, 24), pred.reshape(N, 2, 2, 24), cmaq.reshape(N, 2, 2, 24)
    
    baseline = {bl: np.concatenate([baseline_pred[bl][st] for st in test_stations], axis = 0).reshape(N, 2, 2, 24) for bl in args.baselines}
    

    for i, (sp, levels) in enumerate(zip(['FSPMC', 'O3'], [args.fspmc_levels, args.o3_levels])):
        if len(levels) > 0:
            partition = [(lb, ub) for lb, ub in zip([0] + levels, levels + [np.infty])]
            for j in range(2):
                for lb, ub in partition:
                    mask = np.vectorize(lambda z: lb <= z < ub)(y[:, i, j, :])
                    
                    print(f'{sp} in [{lb}, {ub}), portion {mask.mean() * 100:.1f}%, day {j}:')
                    for m in metrics:
                        cmaq_value = m(pred = cmaq[:, i, j, :][mask], y = y[:, i, j, :][mask])

                        baseline_value_dic = {bl: m(pred = baseline[bl][:, i, j, :][mask], y = y[:, i, j, :][mask]) for bl in args.baselines}
                        value = m(pred = pred[:, i, j, :][mask], y = y[:, i, j, :][mask])
                    
                        print(f'CMAQ, Species {sp}, metric {m.__name__}: {cmaq_value:.4g}')
                        for bl in args.baselines:
                            print(f'Baseline {bl}, Species {sp}, metric {m.__name__}: {baseline_value_dic[bl]:.4g}')
                        print(f'Broadcasting, Species {sp}, metric {m.__name__}: {value:.4g}')
                    print()