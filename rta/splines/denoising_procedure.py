%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture
from collections import Counter

from rta.read_in_data import DT as D
from rta.preprocessing import preprocess
from rta.models.base_model import predict, fitted, coef, residuals
from rta.models import spline
from rta.models.plot import plot, plot_curve

# better use natural splines
formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
DF = preprocess(D, min_runs_no = 2)

# Removing the peptides that have their RT equal to the median.
# TODO: think if these points cannot be directly modelled.
# and what if we start moving them around?
DF = DF[DF.rt_median_distance != 0]

# All points are 'signal' before denoising: guardians
DF['signal'] = 'signal'

# Division by run number
runs = list(data for _, data in DF.groupby('run'))

# data = runs[0]
def denoise_and_align(runs, formula, refit=True):
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


%%time
models = list(denoise_and_align(runs, formula))


d = models[0][1]
m = models[0][0]

%matplotlib
plt.plot(d[d.signal == 'signal'].rt, d[d.signal == 'signal'].rt_aligned)
plt.hist(residuals(m))


%matplotlib
plt.scatter(d.rt,
            d.rt_median_distance,
            c=[{'signal': 'red', 'noise': 'grey'}[s] for s in d.signal])
plot_curve(m, c='blue', linewidth=4)
%matplotlib
plot(models[1][0])






# %matplotlib
# divmod(6,5)
# for g, model in models.items():
#     div, mod = divmod(g, 5)
#     plt.subplot(2, div + 1, mod)
#     plot(model)
#
# plt.subplot(2, 2, 2)
# plot(models[2])
