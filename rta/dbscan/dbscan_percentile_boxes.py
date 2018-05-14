from collections import Counter
from itertools import islice, product, repeat
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn import cluster
import networkx as nx

from rta.misc import max_space
from rta.preprocessing import preprocess
from rta.read_in_data import DT as D
from rta.splines.denoising import denoise_and_align


formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
DF = preprocess(D, min_runs_no = 2)
# Removing the peptides that have their RT equal to the median.
# TODO: think if these points cannot be directly modelled.
# and what if we start moving them around?
# WARNING: skipping this for now.
# DF = DF[DF.rt_median_distance != 0]


# All points are 'signal' before denoising: guardians
DF['signal'] = 'signal'
# Division by run number
runs = list(data for _, data in DF.groupby('run'))
# assembling the models
models = list(denoise_and_align(runs, formula))

# resemble all the data sets
DF_2 = pd.concat(d for m, d in models)

# The noise seems to be confirmed by the analysis of the drift times.
[{g: np.median(np.abs(x.dt_median_distance))
  for g, x in d.groupby('signal')}
 for _, d in models]

[{g: np.median(np.abs(x.le_mass_median_distance))
  for g, x in d.groupby('signal')}
 for _, d in models]

# Trying out the modelling of the percentiles of the data.

# Calculate the spaces in different directions for signal points
DF_2_signal = DF_2[DF_2.signal == 'signal']
# data = DF_2_signal
# DBSCAN_args = {}
def normalize_and_cluster(data_and_percentiles):
    percentiles, data, DBSCAN_args = data_and_percentiles
    NORMALIZED = data.copy()
    for col, percentile in zip(['le_mass', 'rt_aligned', 'dt'],
                               percentiles):
        NORMALIZED[col] = (data[col] - data[col].min()) / percentile
    DBSCAN = cluster.DBSCAN(**DBSCAN_args)
    dbscan_res = DBSCAN.fit(NORMALIZED)
    return dbscan_res

# data = DF_2_signal
# DBSCAN_args = {}
def cluster_percentile_test(data,
                            percentiles     = np.arange(10,101,10),
                            colnames        = ['le_mass', 'rt_aligned', 'dt'],
                            workers_cnt     = 15,
                            eps             = 1.0,
                            min_samples     = 3,
                            metric          = 'chebyshev',
                            **DBSCAN_args):
    """Run the test of percentile clustering."""
    X = data.groupby('id')
    DBSCAN_args.update(dict(metric      = metric,
                            eps         = eps,
                            min_samples = min_samples))
    D_stats = pd.DataFrame(dict(runs_no_aligned      = X.rt.count(),
                                rt_aligned_median    = X.rt.median(),
                                rt_aligned_min       = X.rt.min(),
                                rt_aligned_max       = X.rt.max(),
                                rt_aligned_max_space = X.rt_aligned.aggregate(max_space),
                                pep_mass_max__space  = X.pep_mass.aggregate(max_space),
                                le_mass_max_space    = X.le_mass.aggregate(max_space),
                                le_mass_aligned_min  = X.le_mass.min(),
                                le_mass_aligned_max  = X.le_mass.max(),
                                dt_max_space         = X.dt.aggregate(max_space),
                                dt_aligned_min       = X.dt.min(),
                                dt_aligned_max       = X.dt.max()))

    # Getting the percentiles of selected features
    percentiles = {col: np.percentile(D_stats[col + '_max_space'], percentiles)
                   for col in colnames}
    CLUST_ME = DF_2_signal[colnames]
    data_and_percentiles = zip(product(*[percentiles[name] for name in colnames]),
                               repeat(CLUST_ME),
                               repeat(DBSCAN_args))

    with Pool(workers_cnt) as workers:
        dbscans = workers.map(normalize_and_cluster,
                              data_and_percentiles)

    return zip(product(*[percentiles[name] for name in colnames]),
               dbscans)

dbscans = list(cluster_percentile_test(DF_2_signal))

import pickle

with open('rta/data/dbscans.pickle', 'wb') as h:
    pickle.dump(dbscans, h)
