import numpy as np
import pandas as pd

from rta.models.base_model import predict
from rta.models.SQSpline import SQSpline
from rta.stats.stats import mae, mad, confusion_matrix


def cv_search_iterator(data,
                       parameters,
                       Model=SQSpline,
                       fold_stats=(mae, mad),
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


def tasks_run_param(data, parameters, *other_worker_args):
    folds = np.unique(data.fold)
    for run, d_run in data.groupby('run'):
        d_run = d_run.sort_values('rt')
        d_run = d_run.drop_duplicates('rt')
        for param in parameters:
            out = [run, d_run, param, folds]
            out.extend(other_worker_args)
            yield out

# this looks more like a fit method for the bloody model
# tasks_run_param pipes in parameters for this.
def cv_run_param(run, d_run, param, folds,
                 Model=SQSpline,
                 fold_stats=(mae, mad),
                 model_stats=(np.mean, np.median, np.std)):
    """Cross-validate a model under a given 'run' and 'param'."""
    m = Model()
    m.fit(d_run.rt.values, 
          d_run.rt_median_distance.values,
          **param)
    m_stats = []
    cv_out = []
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
        cv_out.append((n, stats, cm))
    # process stats
    m_stats = np.array(m_stats)
    m_stats = np.array([stat(m_stats, axis=0) for stat in model_stats])
    m_stats = pd.DataFrame(m_stats)
    m_stats.columns = ["fold_" + fs.__name__ for fs in fold_stats]
    m_stats.index = [ms.__name__ for ms in model_stats]

    return run, param, m, m_stats, cv_out

