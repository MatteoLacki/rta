import numpy as np
from sklearn import mixture

from rta.models import spline
from rta.models import predict, fitted, coef, residuals

def denoise_and_align(runs, formula,
                      model='Huber',
                      refit=True):
    """Remove noise in grouped data and align the retention times.

    Updates the 'signal' column in the original data chunks.
    """
    assert all('signal' in run for run in runs)
    for data in runs:
        # Fitting the spline
        signal_indices = data.signal == 'signal'
        model = spline(data[signal_indices], formula)

        # Fitting the Gaussian mixture model
        res = residuals(model).reshape((-1,1))
        gmm = mixture.GaussianMixture(n_components=2,
                                      covariance_type='full').fit(res)

        # The signal is the cluster with smaller variance
        signal_idx, noise_idx = np.argsort(gmm.covariances_.ravel())
        sn = {signal_idx: 'signal', noise_idx: 'noise'}

        # Retaggin some signal tags to noise tags
        data.loc[signal_indices, 'signal'] = [ sn[i] for i in gmm.predict(res)]

        # Refitting the spline only to the signal peptides
        if refit:
            model = spline(data[data.signal == 'signal'], formula)

        # Calculating the new retention times
        data.loc[signal_indices, 'rt_aligned'] = data[data.signal == 'signal'].rt - fitted(model)

        # Coup de grace!
        yield model, data
