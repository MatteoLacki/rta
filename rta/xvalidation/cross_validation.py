"""Cross validate the input model."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

from collections import Counter as Count
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from numpy.random import shuffle
import pandas as pd

from rta.models.base_model import coef, predict, fitted, coefficients, residuals, res, cv
from rta.models.plot import plot
from rta.models.SQSpline import SQSpline
from rta.read_in_data import data_folder, big_data
from rta.preprocessing import preprocess
from rta.xvalidation.grouped_k_folds import grouped_K_folds
from rta.xvalidation.filters import filter_K_foldable

# data = pd.read_csv(data_folder("one_run_5_folds.csv"))
# chunks_no = 20
# s_model = SQSpline()
# s_model.df_2_data(data, 'rt', 'rt_median_distance')
# s_model.fit(chunks_no=chunks_no)

# optmize the creation of these data-sets
annotated, unlabelled = big_data()
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)

# this should be part of preprocessing
folds_no = K = 5
annotated_cv, annotated_stats_cv, run_counts = filter_K_foldable(annotated, annotated_stats, K)
annotated_cv, annotated_stats_cv, run_counts = annotated_cv.copy(), annotated_stats_cv.copy(), run_counts.copy()
annotated_cv_slim = annotated_cv[['id', 'run', 'rt', 'rt_median_distance']]
annotated_cv_slim.reset_index(inplace=True, drop=True)



# TODO: we do not need that: replace it with simple for with if condition.
# what are the numbers of peptides in each foldable group?
peptide_counts_in_runs = dict(((runs, cnt) for runs, cnt in run_counts.items() if cnt >= K))

def test(N, K):
    groups = np.full((N,), 0)
    # produce an array of K numbers repeated 
    N_div_K = N // K
    N_mod_K = N % K
    for i in range(1, K):
        groups[ i * N_div_K : (i+1) * N_div_K ] = i
    if N_mod_K: 
        group_tags = np.arange(K)
        shuffle(group_tags)         # random permutation of groups
        groups[-N_mod_K:] = group_tags[0:N_mod_K] # assigning the remainder
    shuffle(groups)
    return groups

# so now, we have to build up space for the folds indices

def assign_peptides_to_folds(peptides_cnt):
    folds = np.zeros(peptides_cnt, dtype=np.int8)
    s = e = 0
    for cnt in peptide_counts_in_runs.values():
        e += cnt
        folds[s:e] = test(cnt, K)
        s = e
    return folds

# this divides the proteins into folds
annotated_stats_cv['fold'] = assign_peptides_to_folds(len(annotated_stats_cv))

# now, we have to divide proteins in runs into groups
annotated_cv_slim = pd.merge(annotated_cv_slim, 
                             pd.DataFrame(annotated_stats_cv.fold),
                             left_on='id',
                             right_index=True)
annotated_cv_slim.head()

