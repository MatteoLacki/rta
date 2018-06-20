"""Cross validate the input model."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler
from collections import Counter as count
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 4)
#
from rta.models.base_model import coef, predict, fitted, coefficients, residuals, res, cv
from rta.models.plot import plot
from rta.models.SQSpline import SQSpline
from rta.read_in_data import data_folder, big_data
from rta.preprocessing import preprocess
# PARAMETERS
from rta.default_parameters import *
# 
annotated_all, unlabelled_all = big_data()
annotated_cv, annotated_stats, runs_cnts = preprocess(annotated_all,
                                                      min_runs_no,
                                                      folds_no)
slim_features = ['id','run','rt','rt_median_distance','fold']
annotated_cv_slim = annotated_cv[slim_features]
data = annotated_cv_slim

def get_folded_data(data, folds_no):
    AND = np.logical_and
    runs = np.unique(data.run)
    folds = np.arange(folds_no)
    for run in runs:
        for fold in folds:
            train = data.loc[AND(data.run == run, data.fold != fold),:]
            test = data.loc[AND(data.run == run, data.fold == fold),:]
            yield run, train, test




chunks_no = 20
s_model = SQSpline()
s_model.df_2_data(data, 'rt', 'rt_median_distance')
s_model.fit(chunks_no=chunks_no)
