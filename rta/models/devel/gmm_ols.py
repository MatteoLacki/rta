"""Check the distribution of the error terms from one run of Huber regression.

This way we will ascertain, if only one step of Huber fitting is necessary.
Alternatively, we could refit on the 'signal' data with either L2 or L1 regressors.
Possibly checking for the values of some parameters.
"""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrices, dmatrix, bs, cr, cc
from sklearn.linear_model import HuberRegressor
from statsmodels import robust

from rta.models.base_model import cv, coef, predict
from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.xvalidation import grouped_K_folds, filter_foldable
from rta.models.GMM_OLS_combo import GMM_OLS
from rta.models.plot import plot

# get data
annotated, unlabelled = big_data()
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)
K = 5 # number of folds
annotated_cv = filter_foldable(annotated, annotated_stats, K)
folds = annotated_cv.groupby('runs').rt.transform(grouped_K_folds,
                                                  K = K).astype(np.int8)
annotated_cv = annotated_cv.assign(fold=folds)
annotated_cv_1 = annotated_cv[annotated_cv.run == 1]
data = annotated_cv_1
data = data.sort_values(['rt', 'rt_median_distance'])

# chunks_no = 100
# gmm_ols = GMM_OLS()
# formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
# gmm_ols.fit(formula, data, data_sorted=True, chunks_no=chunks_no)



# TODO:
    # reproduce the spline fitting with state-transforms
        # adjust the predict function
    # the fitting of splines can potentially be done repeatedly, so we will need these classes to 
    # accept new arguments. 
        # set warm starts everywhere
        # splines will also need to be recalculated.
# Question: how is beta spline evalution implemented by PATSY? Check source code.
    # it calls scipy's implementation of Bsplines, that is a wrapper around some Fortran code


from collections import Counter as count
from scipy.interpolate import LSQUnivariateSpline as Spline
from sklearn.mixture import GaussianMixture as GM

from rta.models.base_model import Model
from rta.array_operations.misc import percentiles, percentile_pairs_of_N_integers as percentile_pairs

# FIX THIS SHIT!!!


class GMM_OLS(Model):
    def __init__(self):
        self.has_data = False

    def df_2_data(self, data, x_name='x', y_name='y'):
        """Prepare data, if not ready."""
        data = data.drop_duplicates(subset=x_name, keep=False, inplace=False)
        data = data.sort_values([x_name, y_name])
        self.x, self.y = (np.asarray(data[name]).reshape(-1,1) 
                          for name in (x_name, y_name))
        self.has_data = True

    def fit(self,
            x=None,
            y=None,
            chunks_no=100,
            **kwds):
        """Fit a denoised spline."""
        if not self.has_data:
            assert not x is None or not y is None
            self.x, self.y = x, y
        else:
            x, y = self.x, self.y
        N_obs = len(x)
        self.signal = signal = np.empty(N_obs, dtype=np.bool_)
        self.probs = probs = np.empty((chunks_no, 2), dtype=np.float64)
        self.means = means = probs.copy()
        self.covariances = covariances = probs.copy()
        g_mix = GM(n_components = 2, warm_start=False)
        for i, (s, e) in enumerate(percentile_pairs(N_obs, chunks_no)):
            g_mix.fit(x[s:e])
            # signal has smaller variance
            idxs = signal_idx, noise_idx = np.argsort(g_mix.covariances_.ravel())
            signal[s:e] = g_mix.predict(x[s:e]) == signal_idx
            # other chunk-specific outputs
            probs[i,:] = g_mix.weights_[idxs]
            means[i,:] = g_mix.means_.ravel()[idxs]
            covariances[i,:] = g_mix.covariances_.ravel()[idxs]
        x_signal = x[signal].ravel()
        y_signal = y[signal].ravel()
        x_inner_percentiles = percentiles(x_signal, chunks_no, inner=True)
        self.spline = Spline(x_signal, y_signal, x_inner_percentiles)

    def predict(self, x):
        return self.spline(x).reshape(-1, 1)

    def fitted(self):
        return self.spline(self.x.ravel()).reshape(-1, 1)

    @property
    def res(self):
        """Get residuals."""
        return self.y - self.fitted()

    def __repr__(self):
        """Represent the model."""
        #TODO make this more elaborate.
        return "This is a GMM_OLS combo class for super-duper fitting."    


gmm_ols = GMM_OLS()
data_no_dups = data.drop_duplicates(subset='rt', keep=False, inplace=False)
gmm_ols.df_2_data(data, 'rt', 'rt_median_distance')
gmm_ols.fit(chunks_no = 10)

gmm_ols.fitted()
gmm_ols.predict(np.asarray([10, 20]))
gmm_ols.signal.reshape(-1, 1)

from rta.models.plot import plot
plot(gmm_ols)
plt.show()



# OBSERVATION: if that cannot be optimized in C, than I am a silly wig.
# 98% comes from fitting the Gaussian mixture.
# This must do wrong stuff.

spline.get_coeffs()





