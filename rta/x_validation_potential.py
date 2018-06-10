%load_ext autoreload
%autoreload 2
%load_ext line_profiler

from collections import Counter as count
import numpy as np
import pandas as pd

from rta.read_in_data import big_data
from rta.preprocessing import preprocess

annotated, unlabelled = big_data(path = "~/Projects/retentiontimealignment/Data/")
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)
annotated_slim = annotated[['run', 'id',
                                'rt', 'rt_median_distance', 
                                'dt', 'dt_median_distance']]
unlabelled_slim = unlabelled[['run', 
                                'rt', 'dt']]

runs_no = max(annotated_stats.runs_no)
runs_dtype = f"<U{2*runs_no}"

def ordered_str(ints):
    x = list(ints)
    x.sort()
    return "_".join(str(i) for i in x)

def get_present_runs(summarize = True, dtype="<U20"):
    for pept_id, data in annotated.groupby('id'):
        o = ordered_str(data.run)
        if summarize:
            yield o
        else:
            yield np.full((len(data),), o, dtype)

run_participation_cnt = count(get_present_runs())
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


folds_no = 5


# select runs that can be represented by the K folds
foldable_runs = runs[peptides_cnt >= folds_no]
foldable_runs = set(foldable_runs)

# this takes too much time... optimize!!!
run_stratas = []
for x in get_present_runs(summarize = False, dtype=runs_dtype):
    run_stratas.extend(x)

annotated_slim = annotated_slim.assign(runs=run_stratas)
annotated_cv = annotated_slim[np.array(list(x in foldable_runs 
                                            for x in run_stratas))]


# to use predefined split, we fill have to simply pass the indices of groups.
# either use the existing implementation to iterate over the test groups
# or reimplement to have only test groups

# iterate over strata:

# goal: get the division into folds for each stata and run
#       group = id

def stratify():
    for runs, stratum in annotated_cv.groupby('runs'):
        yield runs, stratum

runs, stratum = next(stratify())

### FUCK FUCK FUCK!!!
# "run" in a group don't correspond to "runs" aggregate!





# diff approach: neglect the precise composition; focus: number of runs
print("Sizes of runs_no-strata:\n", count(annotated_stats.runs_no))



