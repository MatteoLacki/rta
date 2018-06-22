"""Cross validate the input model."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler
from collections import Counter as count
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
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

# data = annotated_cv_slim
# Model=SQSpline
# parameters = [ {"chunks_no": n} for n in np.power(2, range(2,8))]
# run, d_run = 1, data.groupby('run').get_group(1)
# param = parameters[0]
# folds = np.unique(data.fold)
# fold = folds[0]
# statistics=(np.mean, np.std, mad, mean_absolute_deviation)

def mae(x):
    return np.mean(np.abs(x))

def mse(x): 
    return np.std(x)

def cv_search(data,
              parameters,
              Model=SQSpline,
              fold_stats=(mae, mse, mad),
              model_stats=(np.mean, np.median, np.std)):
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
            m_stats = []
            for fold in folds:
                train = d_run.loc[d_run.fold != fold,:]
                test  = d_run.loc[d_run.fold == fold,:]
                n = Model()
                n.fit(x=train.rt.values,
                      y=train.rt_median_distance.values,
                      **param)
                errors = np.abs(predict(n, test.rt.values) - test.rt_median_distance.values)
                n_signal = n.is_signal(test.rt, test.rt_median_distance)
                stats = [stat(errors) for stat in fold_stats]
                m_stats.append(stats)
                cm = confusion_matrix(m.signal[d_run.fold == fold], n_signal)
                yield run, param, n, stats, cm, 'cv'
            # process stats
            m_stats = np.array(m_stats)
            m_stats = np.array([stat(m_stats, axis=0) for stat in model_stats])
            m_stats = pd.DataFrame(m_stats)
            m_stats.columns = ["fold_" + fs.__name__ for fs in fold_stats]
            m_stats.index = [ms.__name__ for ms in model_stats]
            yield run, param, m, m_stats, 'model'

# analyze speed
# %lprun -f cv_search list(cv_search(data, parameters))
# %%timeit
# models = list(cv_search(data, parameters))


# make the plots now showing how the grid search improves the metrics
# parameters = [ {"chunks_no": n} for n in np.power(2, range(2,10))]

parameters = [{"chunks_no": n} for n in range(2,50)]
models = list(cv_search(annotated_cv_slim, parameters))

# to make it MP compatible, we have to make it into a series of tasks.
# have processes with subprocesses
# cores_no = 16
# with Pool(cores_no) as p:
#     models = p.map(f, [1, 2, 3]))
models[-1]

models_stats = pd.DataFrame([dict(run         = m[0], 
                                  chunks_no   = m[1]['chunks_no'],
                                  mae_mean    = m[3].loc['mean',    'fold_mae'],
                                  mae_median  = m[3].loc['median',  'fold_mae'],
                                  mae_std     = m[3].loc['std',     'fold_mae'], 
                                  mse_mean    = m[3].loc['mean',    'fold_mse'],
                                  mse_median  = m[3].loc['median',  'fold_mse'],
                                  mse_std     = m[3].loc['std',     'fold_mse'],
                                  mad_mean    = m[3].loc['mean',    'fold_mad'],
                                  mad_median  = m[3].loc['median',  'fold_mad'],
                                  mad_std     = m[3].loc['std',     'fold_mad'])
                             for m in models if m[-1] == 'model'])
models_stats.to_csv(path_or_buf = "~/Desktop/cs_stats.csv", index=False)


models_org = [[]] * len(parameters)
i = 0
for model in models:
    models_org
    mod





models[ 0 ]
models[ 0:10 ]
x = models[ 10 ][ 3 ]
x = pd.DataFrame(x)
x.columns = [fs.__name__ for fs in fold_stats]
x.index = [ms.__name__ for ms in model_stats]



