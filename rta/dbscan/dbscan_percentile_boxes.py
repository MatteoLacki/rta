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

def relabel_noise(labels):
    label = -1
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = label
            label -= 1
    return labels

params_and_labels = [(p, relabel_noise(d.labels_)) for p, d in dbscans]

## This roughly corresponds to the Purity index
# def grouped_ness(d):
#     if any(d.signal):
#         x = Counter(d.cluster[d.signal])
#         return x[max(x)] / len(d.signal)
#     else:
#         return 0
# def get_distribution_of_groupedness_index(x):
#     labels, ids = x
#     small_df = pd.DataFrame({'id': ids, 'cluster': labels})
#     small_df = small_df.assign(signal = small_df.cluster != -1)
#     grouped_ness_idx = [(id, grouped_ness(d)) for id, d in small_df.groupby('id')]
#     return np.histogram(list(b for a, b in grouped_ness_idx),
#                         bins = np.arange(0,1.1,.1))[0]
#
#
#
# get_distribution_of_groupedness_index((dbscans[0][1].labels_, DF_2_signal.id))
# # how can this take sooo much time????
# workers_cnt = 15
# with Pool(workers_cnt) as workers:
#     groupedness = workers.map(get_distribution_of_groupedness_index,
#                               zip((dbscan.labels_ for p, dbscan in dbscans),
#                                   repeat(DF_2_signal.id)))
# groupedness_arr = np.asarray(groupedness)
# percentiles = np.asarray([p for p, d in dbscans])
#
# groupedness_df = pd.DataFrame(np.concatenate((percentiles, groupedness_arr), axis=1))
# groupedness_df.columns = ['le_mass', 'rt_aligned', 'dt'] + list(np.arange(0, 1, .1))
# groupedness_df.to_csv('rta/data/groupedness.csv', index=False)

# getting some proper measures of clustering efficiency:
# the completeness_score and the homogeneity_scores!!!


from sklearn.metrics import completeness_score, homogeneity_score

## anything iterable is accepted
# completeness_score(labels_true = ['1', 'b', 'b', 2],
#                    labels_pred = [1, 2, 2, 3] )


dbscan_labels = dbscans[0][1].labels_
completeness_score(labels_true = DF_2_signal.id,
                   labels_pred = dbscan_labels )

homogeneity_score(labels_true = DF_2_signal.id,
                   labels_pred = dbscan_labels )

homogeneity_score(labels_true = [-1, -2, -3, -4],
                  labels_pred = [1 , 1, 1, 1])


def simplify_indices(prot_ids):
    classes = [0] * len(prot_ids)
    used_names = set()
    label = 0
    for i, p in enumerate(prot_ids):
        if p not in used_names:
            used_names.add(p)
            label += 1
        classes[i] = label
    return classes

classes = simplify_indices(DF_2_signal.id)

# slow version: one process
# scores = pd.DataFrame([p + (completeness_score(prot_ids, d.labels_),
#                             homogeneity_score(prot_ids,  d.labels_))
#                        for p, d in dbscans])

# multiprocess version
workers_cnt = 15

def get_scores(arg):
    p, l, real_l = arg
    return p + (completeness_score(real_l, l), homogeneity_score(real_l, l))

with Pool(workers_cnt) as workers:
    scores = workers.map(get_scores,
                       ((p,d,r) for (p,d), r in zip(params_and_labels,
                                                    repeat(classes))))
scores = pd.DataFrame(scores)


scores.columns = ('le_mass', 'rt_aligned', 'dt', 'completeness', 'homogeneity')
scores.head()

scores.to_csv('rta/data/completeness_homogeneity.csv', index=False)


# Investigating alternative definitions of errors:
p, clusters = params_and_labels[0]
clusterings = [l for d, l in params_and_labels]

# contingency_table = Counter(zip(l, classes))



def Stefan_metrics(classes_clusters):
    classes, clusters = classes_clusters
    X = pd.DataFrame(dict(classes = classes, clusters = clusters))
    c_Stefan = X.groupby('classes')['clusters'].apply(lambda x: len(np.unique(x)))
    c_Stefan = sum(c_Stefan == 1) / len(h_Stefan)
    h_Stefan = X.groupby('clusters')['classes'].apply(lambda x: len(np.unique(x)))
    h_Stefan = sum(h_Stefan == 1) / len(h_Stefan)
    return c_Stefan, h_Stefan

workers_cnt = 15

with Pool(workers_cnt) as workers:
    scores = workers.map(Stefan_metrics,
                         zip(repeat(classes), clusterings))
