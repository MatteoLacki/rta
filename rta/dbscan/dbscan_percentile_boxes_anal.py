import pickle

with open('rta/data/dbscans.pickle', 'rb') as h:
    dbscans = pickle.load(h)

with open('rta/data/DF_2_signal.pickle', 'rb') as h:
    DF_2_signal = pickle.load(h)


# Moving to R.
# DF_2_signal.to_csv(path_or_buf = 'rta/data/denoised_data.csv',
#                    index       = False)
#
# clusterings = pd.DataFrame({ "__".join(str(a) for a in p): d.labels_ for p, d in dbscans})
# clusterings.to_csv(path_or_buf = 'rta/data/clusterings.csv',
#                    index       = False)



# Deriving the criterion for the evaluation of clustering
# TODO: make it more X-validation like

# data = DF_2_signal
# dbscan = dbscans[500][1]
def get_stats(data, dbscan):
    """Calculate the statistics for the labels."""
    data = data.copy()
    data = data.assign(cluster = dbscan.labels_)
    data = data.assign(noise = data.cluster == -1)

    grouped_data = data.groupby('id').noise
    data_stats = pd.DataFrame(dict(noise_cnt  = grouped_data.sum(),
                                   runs_no    = grouped_data.count()))

    data_stats.loc[data_stats.noise_cnt < data_stats.runs_no,]

    data_stats = data_stats.assign(signal     = data_stats.runs_no - data_stats.noise_cnt,
                                   noise_norm = data_stats.noise_cnt / data_stats.runs_no)
    data_stats = data_stats.assign(signal_norm = 1.0 - data_stats.noise_norm)
    final_stats = data_stats.quantile(np.linspace(0.0, 1.0, 11))
    return final_stats, data_stats



# data_stats.loc[(data_stats.signal + data_stats.noise_cnt) != data_stats.runs_no,]

stats = [(params, get_stats(DF_2_signal, dbscan)) for params, dbscan in dbscans]
# small_stats = [f.assign(le_mass = p[0],
#                         rt_aligned = p[1],
#                         dt = p[2]) for p, (f, d) in stats]
small_stats = [p, f for p, (f, _) in stats]

small_stats = pd.concat(small_stats)



small_stats.loc[small_stats.signal == 7,]

small_stats.to_csv(path_or_buf = 'rta/data/percentiles_and_clustering.csv',
                   index       = True)
