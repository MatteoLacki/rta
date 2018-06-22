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
from rta.models.SQSpline import SQSpline, mad, mean_absolute_deviation
from rta.read_in_data import data_folder, big_data
from rta.preprocessing import preprocess
# PARAMETERS
from rta.default_parameters import *
from sklearn.metrics import confusion_matrix

annotated_all, unlabelled_all = big_data()

# this already preprocessed the data.
annotated_cv, annotated_stats, runs_cnts = preprocess(annotated_all,
                                                      min_runs_no,
                                                      folds_no)
slim_features = ['id','run','rt','rt_median_distance','fold']
annotated_cv_slim = annotated_cv[slim_features]


data = annotated_cv_slim
Model=SQSpline

parameters = [ {"chunks_no": n} for n in np.power(2, range(2,8))]
run, d_run = 1, data.groupby('run').get_group(1)
param = parameters[0]
folds = np.unique(data.fold)
fold = folds[0]
statistics=(np.mean, np.std, mad, mean_absolute_deviation)


def cv_search(data,
              parameters,
              Model=SQSpline,
              statistics=(np.mean, np.std, mad, mean_absolute_deviation)):
    folds = np.unique(data.fold)
    for run, d_run in data.groupby('run'):
        d_run = d_run.sort_values('rt')
        d_run = d_run.drop_duplicates('rt')
        # grid search
        for param in parameters:
            m = Model()
            m.fit(d_run.rt.values, 
                      d_run.rt_median_distance.values,
                      **param)
            yield run, param, m
            for fold in folds:
                train = d_run.loc[d_run.fold != fold,:]
                test = d_run.loc[d_run.fold == fold,:]
                n = Model()
                n.fit(x=train.rt.values,
                      y=train.rt_median_distance.values)
                test_pred = predict(n, test.rt.values)
                n_signal = n.is_signal(test.rt, test.rt_median_distance)
                stats = [stat(test_pred) for stat in statistics]
                cm = confusion_matrix(m.signal[d_run.fold == fold], n_signal)
                yield run, param, n, stats, cm

# analyze the data
%lprun -f cv_search list(cv_search(data, parameters))

# make the plots now showing how the grid search improves the metrics
