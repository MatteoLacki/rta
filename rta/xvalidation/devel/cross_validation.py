"""Cross validate the input model."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 4)

from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.xvalidation.cross_validation import tasks_run_param, cv_run_param
from rta.stats.stats import compare_fold_quantiles

annotated_all, unlabelled_all = big_data()

folds_no = 10
min_runs_no = 5
annotated_cv, annotated_stats, runs_cnts = preprocess(annotated_all,
                                                      min_runs_no,
                                                      folds_no)
slim_features = ['id','run','rt','rt_median_distance','fold']
data = annotated_cv_slim = annotated_cv[slim_features]

cores_no = 16
data = annotated_cv_slim
Model = SQSpline
parameters = [{"chunks_no": n} for n in range(2,50)]

with Pool(cores_no) as p:
    results = p.starmap(cv_run_param, tasks_run_param(data, parameters))

# check if the simple x-validation scheme offers good coverage of RT.
data = annotated_cv_slim
compare_fold_quantiles(data)


# developing the other x-validation scheme
D_stats = annotated_stats
D_stats.sort_values("runs", inplace=True)
run_cnts = D_stats.groupby("runs").runs.count()
run_cnts = run_cnts[run_cnts >= folds_no].copy()
D_stats = D_stats.loc[D_stats.runs.isin(run_cnts.index)].copy()
# we need sorted DF to append a column correctly


from rta.xvalidation.stratifications_folds import peptide_stratified_folds as fold
from rta.xvalidation.stratifications_folds import tenzer_folds, iter_tenzer_folds, shuffled_folds



folds = list(range(folds_no))
shuffle(folds)


# D_stats['fold'] = 
fold(peptides_cnt = len(D_stats), run_cnts = run_cnts, folds_no = folds_no)


