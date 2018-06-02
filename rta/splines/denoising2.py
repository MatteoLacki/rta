from multiprocessing import Pool
import numpy as np
import pandas as pd
from pandas import DataFrame as DF
from sklearn import mixture

from rta.models import spline
from rta.models import predict, fitted, coef, residuals


def denoise_and_align_run(annotated_run,
                          unlabelled_run,
                          formula,
                          model = 'Huber',
                          refit = True,
                          return_model = False,
                          **kwds):
    """Remove noise and align the retention times in a run."""
    a, u = annotated_run, unlabelled_run

    # fit the spline
    model = spline(a, formula, **kwds)

    # fit the Gaussian mixture
    res = residuals(model).reshape((-1,1))
    gmm = mixture.GaussianMixture(n_components=2, # only 2: noise & signal
                                  covariance_type='full').fit(res)

    # signal has smaller variance
    signal_idx, noise_idx = np.argsort(gmm.covariances_.ravel())
    signal = np.array([signal_idx == i for i in gmm.predict(res)])

    # refit the spline on the "signal" peptides
    if refit:
        # TODO add x-validation to the model here.
        # this will destroy the rest of the code...
        model = spline(a[signal], formula)

    return 0

# supprisingly, this works!
def denoise_and_align(annotated, unlabelled,
                      formula,
                      model='Huber',
                      refit=True,
                      workers_cnt=16):
    """Denoise and align all runs."""

    def iter_groups():
        for run_no, a in annotated.groupby('run'):
            u = unlabelled[unlabelled.run == run_no]
            yield a, u, formula, model, refit

    with Pool(workers_cnt) as workers:
        res = workers.starmap(denoise_and_align_run, iter_groups())

    return res
