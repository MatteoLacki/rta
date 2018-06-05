import gc
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pandas import DataFrame as DF
from sklearn import mixture

from rta.models import spline
from rta.models import predict, fitted, coef, residuals


def denoise_and_align_run(annotated_run,
                          unnanotated_run,
                          formula,
                          model_name = 'Huber',
                          refit = True,
                          **kwds):
    """Remove noise and align the retention times in a run."""
    a, u = annotated_run, unnanotated_run

    # fit the spline
    model = spline(a, formula, model_name, **kwds)

    # fit the Gaussian mixture
    res = residuals(model).reshape((-1,1))
    gmm = mixture.GaussianMixture(n_components=2, # only 2: noise & signal
                                  covariance_type='full').fit(res)

    # signal has smaller variance
    signal_idx, noise_idx = np.argsort(gmm.covariances_.ravel())
    signal = np.array([signal_idx == i for i in gmm.predict(res)])

    # refit the spline on the "signal" peptides
    if refit:
        model = spline(a[signal], formula)

    a_rt_aligned = np.array(a.rt) - predict(model, rt=a.rt)
    u_rt_aligned = np.array(u.rt) - predict(model, rt=u.rt)

    return signal, a_rt_aligned, u_rt_aligned


# supprisingly, this works!
def denoise(annotated,
            unlabelled,
            formula,
            model_name='Huber',
            refit=True,
            workers_cnt=16):
    """Denoise all runs."""

    def iter_groups():
        for run_no, a in annotated.groupby('run'):
            u = unlabelled[unlabelled.run == run_no]
            yield a, u, formula, model_name, refit

    with Pool(workers_cnt) as workers:
        res = workers.starmap(denoise_and_align_run, iter_groups())

    return res
