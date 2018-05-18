%load_ext autoreload
%autoreload 2

from collections import Counter
from functools import partial
from itertools import islice, product, repeat
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn import cluster
import networkx as nx


from rta.misc import max_space
from rta.preprocessing import preprocess
from rta.read_in_data import big_data
# from rta.splines.denoising import denoise_and_align

annotated_DT, unlabelled_DT = big_data()
annotated_DT, Dstats = preprocess(annotated_DT, min_runs_no = 2)

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
# Removing the peptides that have their RT equal to the median.
# TODO: think if these points cannot be directly modelled.
# and what if we start moving them around?
# WARNING: skipping this for now.
# DF = DF[DF.rt_median_distance != 0]

def denoise_and_align(annotated_DT,
                      unlabelled_DT,
                      formula,
                      model         ='Huber',
                      refit         = True):
    """Remove noise in grouped data and align the retention times in a run.

    Updates the 'signal' column in the original data chunks.
    """
    # All points are 'signal' before denoising: guardians
    annotated_DT['signal'] = 'signal'
    for run, annotated in annotated_DT.groupby('run'):
        unlabelled = unlabelled_DT[unlabelled_DT.run == run]
        yield (annotated,
                unlabelled,
                formula,
                model,
                refit)

results = list(denoise_and_align(annotated_DT,
                                 unlabelled_DT,
                                 formula,
                                 model ='Huber',
                                 refit = True))

import numpy as np
from sklearn import mixture

from rta.models import spline
from rta.models import predict, fitted, coef, residuals

annotated, unlabelledm, formula, model, refitargs = results[0]
# TODO maybe pass only a part of the data frame with the right columns?
def denoise_and_align_run(annotated,
                          unlabelled,
                          formula,
                          model = 'Huber',
                          refit = True):
    """Remove noise in grouped data and align the retention times in a run.

    Updates the 'signal' column in the original data chunks.
    """
    # Fitting the spline
    model = spline(annotated, formula)

    # Fitting the Gaussian mixture model
    res = residuals(model).reshape((-1,1))
    gmm = mixture.GaussianMixture(n_components=2,
                                  covariance_type='full').fit(res)

    # The signal is the cluster with smaller variance
    signal_idx, noise_idx = np.argsort(gmm.covariances_.ravel())
    signal = np.array([signal_idx == i for i in gmm.predict(res)])

    # Refitting the spline only to the signal peptides
    if refit:
        model = spline(annotated[signal], formula)

    # Calculating the new retention times


    annotated.loc[signal_indices, 'rt_aligned'] = \
        annotated[annotated.signal == 'signal'].rt - fitted(model)

    # Aligning the unlabelled data
    # unlabelled.loc[:,'rt_aligned'] =
    unlabelled['rt_aligned'] = unlabelled.rt - predict(model, rt = unlabelled.rt)

    # Coup de grace!
    return annotated, unlabelled, model


def denoise_and_align(annotated_DT,
                      unlabelled_DT,
                      formula,
                      model         ='Huber',
                      refit         = True):
    """Remove noise in grouped data and align the retention times in a run.

    Updates the 'signal' column in the original data chunks.
    """
    # All points are 'signal' before denoising: guardians
    annotated_DT['signal'] = 'signal'
    for run, annotated in annotated_DT.groupby('run'):
        unlabelled = unlabelled_DT[unlabelled_DT.run == run]
        yield (annotated,
                unlabelled,
                formula,
                model,
                refit)
        # yield denoise_and_align_run(annotated,
        #                             unlabelled,
        #                             formula,
        #                             model,
        #                             refit)

args = next(denoise_and_align(annotated_DT, unlabelled_DT, formula))

# resemble all the data sets
DF_2 = pd.concat(d for r, m, d in models)

# apply the models' predictions to the unlabelled_DT
unlabelled_DT

# The noise seems to be confirmed by the analysis of the drift times.
[{g: np.median(np.abs(x.dt_median_distance))
  for g, x in d.groupby('signal')}
 for _, d in models]

[{g: np.median(np.abs(x.mass_median_distance))
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
# scores = pd.DataFrame(scores)
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
    X = pd.DataFrame(((cs, cr, n) for (cs, cr), n in X.items()))
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
scores = pd.DataFrame(scores)
scores.columns = ('c_Stefan', 'h_Stefan', 'c_weighted', 'h_weighted')

params = pd.DataFrame([p for p, l in params_and_labels])
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
    X = pd.DataFrame(((cs, cr, n) for (cs, cr), n in X.items()))
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
swansons = pd.DataFrame(swansons)
swansons.columns = ('swanson', 'swanson_weighted')

swansons = pd.concat([params, swansons], axis=1)
swansons.to_csv('rta/data/swansons_clustering_metrics.csv', index=False)

# validation
p, clusters = params_and_labels[998]

params_and_labels[998][0]
params_and_labels[999][0]
