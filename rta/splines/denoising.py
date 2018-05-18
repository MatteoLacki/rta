import numpy as np
from sklearn import mixture

from rta.models import spline
from rta.models import predict, fitted, coef, residuals


def denoise_and_align_run(annotated,
                          unlabelled,
                          formula,
                          model = 'Huber',
                          refit = True):
    """Remove noise in grouped data and align the retention times in a run.

    Updates the 'signal' column in the original data chunks.
    """
    # Fitting the spline
    model = spline(annotated, formula)

    # Fitting the Gaussian mixture model
    res = residuals(model).reshape((-1,1))
    gmm = mixture.GaussianMixture(n_components=2,
                                  covariance_type='full').fit(res)

    # The signal is the cluster with smaller variance
    signal_idx, noise_idx = np.argsort(gmm.covariances_.ravel())
    signal_indices = np.array([sn[i]== signal_idx for i in gmm.predict(res)])

    # Refitting the spline only to the signal peptides
    if refit:
        model = spline(annotated[signal_indices], formula)

    # Calculating the new retention times
    annotated.loc[signal_indices, 'rt_aligned'] = \
        annotated[annotated.signal == 'signal'].rt - fitted(model)

    # Aligning the unlabelled data
    # unlabelled.loc[:,'rt_aligned'] =
    unlabelled['rt_aligned'] = unlabelled.rt - predict(model, rt = unlabelled.rt)

    # Coup de grace!
    return annotated, unlabelled, model


def denoise_and_align(annotated_DT,
                      unlabelled_DT,
                      formula,
                      model         ='Huber',
                      refit         = True):
    """Remove noise in grouped data and align the retention times in a run.

    Updates the 'signal' column in the original data chunks.
    """
    # All points are 'signal' before denoising: guardians
    annotated_DT['signal'] = 'signal'
    for run, annotated in annotated_DT.groupby('run'):
        unlabelled = unlabelled_DT[unlabelled_DT.run == run]
        yield   annotated,
                unlabelled,
                formula,
                model,
                refit
        # yield denoise_and_align_run(annotated,
        #                             unlabelled,
        #                             formula,
        #                             model,
        #                             refit)
