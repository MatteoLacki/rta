"""Gaussian Mixture Models and Ordinary Least squares.

An alternative to Huber regression.
Divide the X argument into percentiles.
On each group fit a GMM.
Obtain signal/noise and their variances.
Then fit OLS with a diagonal covariance matrix.
"""

import numpy as np
from patsy import dmatrices, ModelDesc
from sklearn.mixture import GaussianMixture as GM

from rta.array_operations.misc import percentile_pairs_of_N_integers as percentile_pairs
from rta.models.spline_regression import SplineRegression
from rta.patsy_operations.parsers import parse_formula


# TODO: warm start?
# not so complicated to merge it if done from top to down
# the iteration of grid search should go from top to bottom
# 
# TODO: Grid search not so difficult in terms of GMMs.
class GMM_OLS(SplineRegression):
    def fit(self,
            formula,
            data={},
            data_sorted=False,
            chunks_no=100,
            weighted=False,
            **kwds):
        """Fit gaussian mixtures to chunks and then OLS."""
        self.formula = formula
        names = C_name, R_name = self.control_name, self.response_name = parse_formula(formula)

        if not data_sorted: # we can avoid that up the call tree
            data = data.sort_values(by = list(names))
        self.data = data
        self.response = R = np.asarray(data[R_name]).reshape(-1,1)
        self.control = C = np.asarray(data[C_name]).reshape(-1,1)

        # here we might need np.asarray -> f****** pickle
        y, X = dmatrices(formula, data)
        self.y = y = y.ravel()
        self.X = X

        N_obs = len(C)
        g_mix = GM(n_components = 2, warm_start=True)
        signal = np.empty(N_obs, dtype=np.bool_)
        self.probs = np.empty((chunks_no, 2), dtype=np.float64)
        self.means = self.probs.copy()
        self.covariances = self.probs.copy()

        # fitting gaussian mixture models on successive chunks of response
        for i, (s, e) in enumerate(percentile_pairs(N_obs, chunks_no)):
            g_mix.fit(R[s:e])

            # signal has smaller variance
            idxs = signal_idx, noise_idx = np.argsort(g_mix.covariances_.ravel())
            signal[s:e] = g_mix.predict(R[s:e]) == signal_idx
            
            # other chunk-specific outputs
            self.probs[i,:] = g_mix.weights_[idxs]
            self.means[i,:] = g_mix.means_.ravel()[idxs]
            self.covariances[i,:] = g_mix.covariances_.ravel()[idxs]

        # fitting linear regression to points considered noise
        if weighted:
            weights = np.exp( - .5 * np.log(self.covariances)).ravel()
            wX = X.copy()
            wy = y.copy()
            for w, (s,e) in zip(weights, percentile_pairs(len(y), chunks_no)):
                wX[s:e] *= w
                wy[s:e] *= w
            self.ols_res = np.linalg.lstsq(a=wX[signal,], b=wy[signal,])
        else:
            self.ols_res = np.linalg.lstsq(a=X[signal,], b=y[signal,])
        self.coef = self.ols_res[0]
        self.signal = signal.reshape(-1,1)

    def __repr__(self):
        return "GMM_OLS"


# some tests
if __name__ == '__main__':
    gmm_ols = GMM_OLS()
    gmm_ols.fit(formula, data, data_sorted=True, chunks=100)

    from rta.models.plot import plot
    plot(gmm_ols)
    plt.show()