%load_ext autoreload
%autoreload 2

from collections import Counter
from functools import partial
from itertools import islice, product, repeat
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pandas import DataFrame as DF
from sklearn import cluster
import networkx as nx


from rta.misc import max_space
from rta.preprocessing import preprocess
from rta.read_in_data import big_data
from rta.splines.denoising import denoise_and_align

annotated, unlabelled = big_data()
annotated, annotated_stats = preprocess(annotated,
                                        min_runs_no=2)

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
res = denoise_and_align(annotated, unlabelled, formula)

# Calculate the spaces in different directions for signal points
DF_2_signal = res[res.status == 'signal']


#TODO write a cpython routine so that it accepts a box with any shape and
# is still quick.
def normalize_and_cluster(data_and_percentiles):
    """Normalize the data and cluster the points.

    The DBSCAN's chebyshev box is a cube.
    We cannot change the box size without making the calculations much slower.
    Thus, we need to scale the data, so that the box size could be set to 1.
    """
    percentiles, data, DBSCAN_args = data_and_percentiles
    NORMALIZED = data.copy()
    for col, percentile in zip(['mass', 'rt_aligned', 'dt'],
                               percentiles):
        NORMALIZED[col] = (data[col] - data[col].min()) / percentile
    DBSCAN = cluster.DBSCAN(**DBSCAN_args)
    dbscan_res = DBSCAN.fit(NORMALIZED)
    return dbscan_res



def cluster_percentile_test(data,
                            percentiles     = np.arange(10,101,10),
                            workers_cnt     = 15,
                            eps             = 1.0,
                            min_samples     = 3,
                            metric          = 'chebyshev',
                            **DBSCAN_args):
    """Run the test of percentile clustering."""
    DBSCAN_args.update(dict(metric = metric, eps=eps, min_samples=min_samples))

    # get statistics
    X = data.groupby('id')
    stats = DF(dict(runs_no_aligned      = X.rt.count(),
                    rt_aligned_median    = X.rt.median(),
                    rt_aligned_min       = X.rt.min(),
                    rt_aligned_max       = X.rt.max(),
                    rt_aligned_max_space = X.rt_aligned.aggregate(max_space),
                    mass_max_space       = X.mass.aggregate(max_space),
                    mass_aligned_min     = X.mass.min(),
                    mass_aligned_max     = X.mass.max(),
                    dt_max_space         = X.dt.aggregate(max_space),
                    dt_aligned_min       = X.dt.min(),
                    dt_aligned_max       = X.dt.max()))

    # get percentiles of
    colnames = ['mass', 'rt_aligned', 'dt']
    percentiles = {col: np.percentile(stats[col + '_max_space'],
                                      percentiles)
                   for col in colnames}
    CLUST_ME = data[colnames]
    data_and_percentiles = zip(product(*[percentiles[name] for name in colnames]),
                               repeat(CLUST_ME),
                               repeat(DBSCAN_args))

    with Pool(workers_cnt) as workers:
        dbscans = workers.map(normalize_and_cluster,
                              data_and_percentiles)

    dbscans = zip(product(*[percentiles[name] for name in colnames]),
                  dbscans)
    dbscans = list(dbscans)
    return dbscans, stats



dbscans, stats = cluster_percentile_test(DF_2_signal)
stats




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
#     small_df = DF({'id': ids, 'cluster': labels})
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
# groupedness_df = DF(np.concatenate((percentiles, groupedness_arr), axis=1))
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
                   labels_pred = dbscan_labels)

# homogeneity_score(labels_true = DF_2_signal.id,
#                    labels_pred = dbscan_labels )
#
# homogeneity_score(labels_true = [-1, -2, -3, -4],
                  # labels_pred = [1 , 1, 1, 1])

def simplify_indices(prot_ids):
    classes = [0] * len(prot_ids)
    used_names = {}
    free_label = 0
    for i, p in enumerate(prot_ids):
        if p not in used_names:
            used_names[p] = free_label
            free_label += 1
        classes[i] = used_names[p]
    return classes

simple_classes = simplify_indices(DF_2_signal.id)
classes = np.array(DF_2_signal.id.copy())

# multiprocess version
# workers_cnt = 15
#
# def get_scores(arg):
#     p, l, real_l = arg
#     return p + (completeness_score(real_l, l), homogeneity_score(real_l, l))
#
# with Pool(workers_cnt) as workers:
#     scores = workers.map(get_scores,
#                        ((p,d,r) for (p,d), r in zip(params_and_labels,
#                                                     repeat(classes))))
# scores = DF(scores)
#
#
# scores.columns = ('le_mass', 'rt_aligned', 'dt', 'completeness', 'homogeneity')
# scores.head()
#
# scores.to_csv('rta/data/completeness_homogeneity.csv', index=False)


## The Stefan's indices.

p, clusters = params_and_labels[999]

clusterings = [l for d, l in params_and_labels]


classes_clusters = simple_classes, clusters

def Stefan_metrics(classes_clusters):
    X = Counter(zip(*classes_clusters))
    classes_cnt, clusters_cnt = [len(np.unique(x)) for x in classes_clusters]
    X = DF(((cs, cr, n) for (cs, cr), n in X.items()))
    X.columns = ('class', 'cluster', 'cnt')
    N = X.cnt.sum()
    C = X.groupby('class').filter(lambda x: x.shape[0]==1)
    c_Stefan = len(C) / classes_cnt
    c_weighted = sum(C.cnt) / N
    H = X.groupby('cluster').filter(lambda x: x.shape[0]==1)
    h_Stefan = len(H) / clusters_cnt
    h_weighted = sum(H.cnt) / N
    return c_Stefan, h_Stefan, c_weighted, h_weighted

workers_cnt = 15

with Pool(workers_cnt) as workers:
    scores = workers.map(Stefan_metrics,
                         zip(repeat(classes), clusterings))

# TODO:  write them down to csv
scores = DF(scores)
scores.columns = ('c_Stefan', 'h_Stefan', 'c_weighted', 'h_weighted')

params = DF([p for p, l in params_and_labels])
params.columns = ('le_mass', 'rt_aligned', 'dt')
scores = pd.concat([params, scores], axis=1)
scores.to_csv('rta/data/completeness_homogeneity_stefan.csv', index=False)


# params.le_mass.max()
# params.rt_aligned.max()
# params.dt.max()

## Full Ron Swanson
[l for d, l in params_and_labels]
classes_clusters = simple_classes, clusters

def Swanson_metrics(classes_clusters):
    """This is a unidimensional metric: just find all crosses"""
    X = Counter(zip(*classes_clusters))
    classes_cnt, clusters_cnt = [len(np.unique(x)) for x in classes_clusters]
    X = DF(((cs, cr, n) for (cs, cr), n in X.items()))
    X.columns = ('class', 'cluster', 'cnt')
    N = X.cnt.sum()
    classes_clusters_pairs_cnt = len(X)
    X = X.groupby('class').filter(lambda x: x.shape[0]==1)
    X = X.groupby('cluster').filter(lambda x: x.shape[0]==1)
    swanson = len(X) / classes_clusters_pairs_cnt
    swanson_weighted = sum(X.cnt) / N
    return swanson, swanson_weighted


workers_cnt = 15
with Pool(workers_cnt) as workers:
    swansons = workers.map(Swanson_metrics,
                           zip(repeat(simple_classes), clusterings))

# TODO:  write them down to csv
swansons = DF(swansons)
swansons.columns = ('swanson', 'swanson_weighted')

swansons = pd.concat([params, swansons], axis=1)
swansons.to_csv('rta/data/swansons_clustering_metrics.csv', index=False)

# validation
p, clusters = params_and_labels[998]

params_and_labels[998][0]
params_and_labels[999][0]
