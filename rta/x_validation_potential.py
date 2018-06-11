%load_ext autoreload
%autoreload 2
%load_ext line_profiler

# Possible optimizations:
    # make sequence & modification & run a multi-index.
    # aggregate accordingly w.r.t. the other dimensions.
    # assign dtypes other than "python objects"? 


from collections import Counter as count
import numpy as np
import numpy.random as 
import pandas as pd
from sklearn.model_selection import cross_val_score, PredefinedSplit

from rta.read_in_data import big_data
from rta.preprocessing import preprocess

path = "~/Projects/retentiontimealignment/Data/"
annotated, unlabelled = big_data(path=path)
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)

# annotated_slim = annotated[['run', 'id',
#                                 'rt', 'rt_median_distance', 
#                                 'dt', 'dt_median_distance']]
# unlabelled_slim = unlabelled[['run', 
#                                 'rt', 'dt']]

runs_no = max(annotated_stats.runs_no)
runs_dtype = f"<U{2*runs_no}"


def get_present_runs(summarize = True, dtype="<U20"):
    for pept_id, data in annotated.groupby('id'):
        o = ordered_str(data.run)
        if summarize:
            yield o
        else:
            yield np.full((len(data),), o, dtype)

run_participation_cnt = count(annotated_stats.runs)
print(len(run_participation_cnt)) # 1013 / 1024 - almost all possibilities!

# make a plot summarizing the measure concentration in the case of 
# common runs numbers.
run_participation_cnt = sorted(run_participation_cnt.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
runs, peptides_cnt = map(np.array, zip(*run_participation_cnt))
probs_optimal_p_sets = np.cumsum(peptides_cnt)
probs_optimal_p_sets = probs_optimal_p_sets/probs_optimal_p_sets[-1]

# the concentration of peptide presence across runs
import matplotlib.pyplot as plt
plt.vlines(range(1,len(probs_optimal_p_sets)+1), [0], probs_optimal_p_sets)
plt.show()
# there are heavy tails here


for folds_no in range(3,11):
    print(f"Percentage of things that can be nicely cast into {folds_no} folds:")
    print(probs_optimal_p_sets[peptides_cnt > folds_no][-1] * 100, '%')

# Now, it's time to run the CV 
    # which CV?
        # Runs seperately every of the 2**10 categories in 5 folds.
        # this way every run will have similar representation of points!
        # disregard points that cannot be cast into folds.
             # what the CV does with them anyway?
    # then, divide each fold into seperate runs and feed the models

def grouped_K_folds(x, K):
    """Get K-folds that respects the inner groupings.

    Divide the elements of "x" into "K" folds.
    Some elements of "x" might repeat.
    Repeating elements must be assigned together to one of the folds.

    Args:
        x (iterable) : x to divide
        K (int) : number of folds

    Return:
        out (dict) : mapping between unique x and their 
    """
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
        npr.shuffle(group_tags)
        groups[-N_mod_K:] = group_tags[0:N_mod_K]
    npr.shuffle(groups)
    id_2_group = dict(zip(ids, groups))
    return np.array([id_2_group[x] for x in x], dtype=np.int8) 


K = 5 # number of folds
# select runs that can be represented by the K folds
foldable_runs = set(runs[peptides_cnt >= K])
annotated_cv = annotated[ annotated.runs.isin(foldable_runs) ]
folds = annotated_cv.groupby('runs').rt.transform(grouped_K_folds, K=K).astype(np.int8)
count(folds)

annotated_cv = annotated_cv.assign(fold=folds)
# ATTENTION sizes of folds withing groups should be similar or equal
for _, d in annotated_cv.groupby("run"):
    a = count(d.fold)
    print(sorted(list(a.items())))


ps = PredefinedSplit(d.fold)
ps.get_n_splits()

for train_index, test_index in ps.split():
   print("TRAIN:", d.loc[train_index], "TEST:", d.loc[test_index])




cross_val_score(estimator, X, CV)


ps.get_n_splits()

print(ps)       

for train_index, test_index in ps.split():



# diff approach: neglect the precise composition; focus: number of runs
print("Sizes of runs_no-strata:\n", count(annotated_stats.runs_no))



