%load_ext autoreload
%autoreload 2
%load_ext line_profiler

# Possible optimizations:
    # make sequence & modification & run a multi-index.
    # aggregate accordingly w.r.t. the other dimensions.
    # assign dtypes other than "python objects"? 


from collections import Counter as count
import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn.model_selection import cross_val_score, PredefinedSplit

from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.models.sklearn_regressors import SklearnRegression

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
