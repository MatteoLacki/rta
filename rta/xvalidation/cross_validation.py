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

data = pd.read_csv(data_folder("one_run_5_folds.csv"))
chunks_no = 20
s_model = SQSpline()
s_model.df_2_data(data, 'rt', 'rt_median_distance')
s_model.fit(chunks_no=chunks_no)

annotated, unlabelled = big_data()
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)

folds_no = K = 5

# only 50 ms
annotated_cv, annotated_stats_cv, run_counts = filter_K_foldable(annotated, annotated_stats, K)

# build up the groups
annotated_stats_cv

# so now, we have to build up space for the folds indices
folds = np.zeros(len(annotated_stats_cv))


# TODO: we do not need that: replace it with simple for with if condition.
# what are the numbers of peptides in each foldable group?
peptide_counts_in_runs = dict(((runs, cnt) for runs, cnt in run_counts.items() if cnt >= K))

# within each "runs" group, do the assignment to folds
annotated_stats_cv.groupby('runs')


# this can be very well done based on the 'run_counts'




def test(x, K):
    N = len(x)
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

folds = annotated_stats_cv.groupby('runs').runs_no.transform(test, K=K)

annotated_cv = pd.merge(annotated_cv,
                        pd.DataFrame(folds),
                        left_on="id",
                        right_index=True)
# hahaha direct pass the series!!!
for group, data in annotated_cv.id.groupby(folds):
    print(group, data)


# TODO : can one group by a value of a function?
# then we could at least save on appending some stupid column.






# we have filled the folds with proper values
# now, we have to order the annotated_cv based on the values of this vector.
# this can be done wihout appending the function

# the order is wrong





# then, we will have to scan the df for the order of these things and pass the views of
# the data to fitting procedures





# will it be faster if we change the whole peptide_id to integers?
annotated_cv.id = annotated_cv.id.astype('category')
annotated_cv.id.dtypes.categories

folds = annotated_cv.groupby('runs').rt.transform(grouped_K_folds, 
                                                  K=K).astype(np.int8)
annotated_cv = annotated_cv.assign(fold=folds)

# implement the whole procedure without data copying.
# simply shuffle the whole DF and then pass the indices marking the 
# beginnings and ends of the data.

# maybe also assure that the data is stored in a continguous way.
# maybe the data should consist of two float entries?

# there should be some initial ordering of the data
annotated_cv.sort_values(by=["run", "rt"], inplace=True)
annotated_cv.run
annotated_cv.rt

x = annotated_cv.id.copy()
ids = np.unique(x)
N = len(ids)
N_div_K = N // K
N_mod_K = N % K
groups = np.full((N,), 0)


for i in range(1, K):
    groups[ i * N_div_K : (i+1) * N_div_K ] = i
if N_mod_K: 
    # we decide at random (sampling without repetition)
    # to which groups we should assign the remaining free indices.
    group_tags = list(range(K))
    shuffle(group_tags)         # random permutation of groups
    groups[-N_mod_K:] = group_tags[0:N_mod_K] # assigning the remainder
shuffle(groups) #???

folds_no = K = 10
grouped_K_folds(annotated_cv.id, 10)
N = len(annotated_stats)






def random_fold_split(N, K):
    N_div_K = N // K
    N_mod_K = N % K
    enlarged_groups = npr.choice(K, N_mod_K, replace=True)
    yield 0
    for 






# the filtering should be part of the statistics procedure
x = np.arange(N)
npr.shuffle(x)

# we have to divide the above vector to smaller chunks
# we have to map numbers to individual ids. 
# but that's easy, since it's simply having to have an array of peptide-ids.


annotated_stats.sort_values('runs', inplace=True)
Count(annotated_stats.runs)

def get_shuffling(x):
    return shuffle(np.arange(len(x)))

annotated_stats.groupby('runs').transform(get())


