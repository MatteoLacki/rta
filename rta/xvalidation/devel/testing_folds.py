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
from rta.xvalidation.cross_validation import tasks_run_param, cv_run_param
from rta.xvalidation.stratifications_folds import tenzer_folds, randomized_tenzer_folds



annotated_all, unlabelled_all = big_data()
folds_no    = 10
min_runs_no = 5


annotated_cv_tf, annotated_stats_tf, runs_cnts_tf = \
    preprocess(annotated_all,
               min_runs_no,
               folds_no,
               tenzer_folds)

real_perc_tz, *fold_stats_tz = compare_fold_quantiles(annotated_cv_tf)
for d, n in zip(fold_stats_tz,
                ['fold_perc', 'fold_perc_stats_by_run', 'fold_perc_stats']):
    d.to_csv(os.path.join("~/Desktop/tmp/tenzer_folds", n + ".csv"))


annotated_cv_rtf, annotated_stats_rtf, runs_cnts_rtf = \
    preprocess(annotated_all,
               min_runs_no,
               folds_no,
               randomized_tenzer_folds)

real_perc_rtf, *fold_stats_rtf = compare_fold_quantiles(annotated_cv_rtf)
for d, n in zip(fold_stats_rtf,
                ['fold_perc', 'fold_perc_stats_by_run', 'fold_perc_stats']):
    d.to_csv(os.path.join("~/Desktop/tmp/randomized_tenzer_folds", n + ".csv"))

#TODO: 
# a plot that will compare folds for a given run

