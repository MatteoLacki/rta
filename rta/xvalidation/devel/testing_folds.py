"""Cross validate the input model."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

from collections import Counter as count
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 4)

from rta.models.base_model import coef, predict, fitted, coefficients, residuals
from rta.models.plot import plot
from rta.models.SQSpline import SQSpline
from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.stats.stats import compare_fold_quantiles
from rta.stats.stats import fold_similarity
from rta.xvalidation.cross_validation import tasks_run_param, cv_run_param
from rta.xvalidation.stratifications_folds import tenzer_folds
from rta.xvalidation.stratifications_folds import randomized_tenzer_folds
from rta.xvalidation.stratifications_folds import replacement_sampled_folds
from rta.xvalidation.stratifications_folds import no_runs_strata_randomized_tenzer_folds

annotated_all, unlabelled_all = big_data()
folds_no    = 10
min_runs_no = 5

annotated_cv_tf, annotated_stats_tf, run_cnts_tf = \
    preprocess(annotated_all,
               min_runs_no,
               folds_no,
               tenzer_folds)

fs = fold_similarity(annotated_cv_tf)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(fs[1])



no_runs_strata_tenzer_folds(run_cnts_tf, folds_no)
no_runs_strata_tenzer_folds([3,4,5], 3)
no_runs_strata_randomized_tenzer_folds(run_cnts_tf, folds_no)
no_runs_strata_randomized_tenzer_folds([3,4,5], 3)

real_perc_tz, *fold_stats_tz = compare_fold_quantiles(annotated_cv_tf)
for d, n in zip(fold_stats_tz,
                ['fold_perc', 'fold_perc_stats_by_run', 'fold_perc_stats']):
    d.to_csv(os.path.join("~/Desktop/tmp/tenzer_folds", n + ".csv"))


annotated_cv_rtf, annotated_stats_rtf, run_cnts_rtf = \
    preprocess(annotated_all,
               min_runs_no,
               folds_no,
               randomized_tenzer_folds)

real_perc_rtf, *fold_stats_rtf = compare_fold_quantiles(annotated_cv_rtf)
for d, n in zip(fold_stats_rtf,
                ['fold_perc', 'fold_perc_stats_by_run', 'fold_perc_stats']):
    d.to_csv(os.path.join("~/Desktop/tmp/randomized_tenzer_folds", n + ".csv"))

# Testing out the procedure:

# one thing:
list(tenzer_folds([4, 5, 6], 10))
list(tenzer_folds([4, 5, 6], 10, shuffle=True))
# so it makes some sense for the small groups to get the shuffling.
# e.g. for groups of peptides that are smaller than the number of folds.
# also, the last remaining proteins generate some inequality consistently

# implement sampling with repetition fold division for comparison
runs_cnts = runs_cnts_tf
# this approach will totally neglect the strata?
# why not.


# ssr = simple sampling with replacement
annotated_cv_ssr, annotated_stats_ssr, run_cnts_ssr = \
    preprocess(annotated_all,
               min_runs_no,
               folds_no,
               replacement_sampled_folds)

real_perc_ssr, *fold_stats_ssr = compare_fold_quantiles(annotated_cv_ssr)
for d, n in zip(fold_stats_ssr,
                ['fold_perc', 'fold_perc_stats_by_run', 'fold_perc_stats']):
    d.to_csv(os.path.join("~/Desktop/tmp/simple_sampling_with_replacement", n + ".csv"))
# OK, this really introduces a lot of additional variability.
fs = fold_similarity(data)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(fs[0])
    # print(fs[1])
    print(fs[2])


