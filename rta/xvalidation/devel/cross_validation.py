"""Cross validate the input model."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler
from collections import Counter as count
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 4)
#
from rta.models.base_model import coef, predict, fitted, coefficients, residuals
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

# obs: fold is a perfect type of column that we could save in data frame
# as it needs to be computed only once and the computation is highly random
def run_fold_training_test(data, folds_no):
    """Works on any data frame with columns run and fold."""
    AND = np.logical_and
    runs = np.unique(data.run)
    folds = np.arange(folds_no)
    for run in runs:
        for fold in folds:
            train = data.loc[AND(data.run == run, data.fold != fold),:]
            train = train.drop_duplicates('rt')
            train.sort_values('rt', inplace=True)
            test = data.loc[AND(data.run == run, data.fold == fold),:]
            yield run, fold, train, test

run, fold, train, test = next(run_fold_training_test(data, folds_no))


chunks_no = 20
model = SQSpline()
model.fit(chunks_no=chunks_no,
          x=train.rt.values,
          y=train.rt_median_distance.values)
pred = predict(model, test.rt.values)
res = pred - test.rt_median_distance.values
res.mean()
res.std()

# we should somehow apply the filiter to other data.
# encode that possibility tomorrow
res[model.signal] # wrong dimension of course


# this could be placed in the bloody base_model
def cv(model, statistics=[np.mean, np.std],**kwds):
    """Cross validate a model."""
    for r, f, train, test in run_fold_training_test():
        model.fit(x=train.rt.values,
                  y=train.rt_median_distance.values)
        predict(model, test.rt.values)