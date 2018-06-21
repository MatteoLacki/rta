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
            train = train.sort_values('rt', inplace=False) # inplace much slower!
            test_mask = AND(data.run == run, data.fold == fold)
            test = data.loc[test_mask,:]
            yield run, fold, train, test, test_mask.values

chunks_no = 20
model = SQSpline()
model.df_2_data(data, 'rt', 'rt_median_distance')
model.fit(x=model.data.rt.values,
          y=model.data.rt_median_distance.values, 
          chunks_no=chunks_no)

# the idea: run the model on whole of the data.
# check for its consistency in applying the threshold
# this could check model's stability
# but so does MSE and MAE
# this is obviously silly, because the most stable model would simply predict
# constantly the same value.
# this could be placed in the bloody base_model

# we might need the original model to return the noise-signal tags.
# check if the Fortran model has more parameters to change.

from sklearn.metrics import confusion_matrix
r, f, train, test, test_mask = next( run_fold_training_test(model.data, folds_no) )
test_mask.shape

def cv(model,
       folds_no=10,
       statistics=(np.mean, np.std, mad, mean_absolute_deviation),
       **kwds):
    """Cross validate a model."""
    orig_signal = model.signal.copy().ravel()
    for r, f, train, test, test_mask in run_fold_training_test(model.data, folds_no):
        model.fit(x=train.rt.values,
                  y=train.rt_median_distance.values)
        test_pred = predict(model, test.rt.values)
        out = [stat(test_pred) for stat in statistics]
        curr_signal = model.is_signal(test.rt.values, test.rt_median_distance.values)
        out.append(confusion_matrix(orig_signal[test_mask], curr_signal))
        yield out

orig_signal.shape
test_mask.shape


%%timeit
cv_stats = list(cv(model, data, folds_no))


%lprun -f run_fold_training_test list(cv(model, data, folds_no))
%lprun -f cv list(cv(model, data, folds_no))