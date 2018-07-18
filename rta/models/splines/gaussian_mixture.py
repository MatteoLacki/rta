"""Gaussian mixture based denoising. """

import numpy as np

from rta.array_operations.misc import percentiles, percentile_pairs_of_N_integers as percentile_pairs
from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.denoising.window_based import sort_by_x
from rta.models.splines.spline import Spline
from rta.models.splines.beta_splines import beta_spline
from rta.models.mixtures.two_component_gaussian_mixture import TwoComponentGaussianMixture as GM


def fit_interlapping_mixtures(x, y,
                              chunks_no  = 20,
                              warm_start = True,
                              sort       = True):
    """Filter based on two-components gaussian models.

    Divides m/z values and intensities into chunks.
    For each chunk, estimate the boundary between noise and signal.
    The estimation applies a sliding window approach based on 3 chunks.
    For example, if chunks_no = 5 and E stands for set that takes part in estimation, F is the 
    set on which we fit, and N is a set not taken into consideration,
    then subsequent fittings for 5-chunk division look like this:
    FENNN, EFENN, NEFEN, NNEFE, NNNEF.

    Args:
        x (np.array) 1D control
        y (np.array) 1D response
        chunks_no (int) The number of quantile bins.
        warm_start (logical): should consecutive gaussian mixture models start from the previously estimated values of paramters. Speeds up computations, but sometimes produces nonsensical values.
        sd_cnt (float) How many standard deviations are considered to be within signal range.
        x_sorted (logical) Are 'x' and 'y' sorted with respect to 'x'.
    Returns:
        signal (np.array of logical values) Is the point considered to be a signal?
        medians (np.array) Estimates of medians in consecutive bins.
        stds (np.array) Estimates of standard deviations in consecutive bins.
        x_percentiles (np.array) Knots of the spline fitting, needed to filter out noise is 'is_signal'.        
    """
    x, y = sort_by_x(x, y) if sort else (x, y)
    signal = np.empty(len(x), dtype = np.bool_)
    g_mix = GM(warm_start = warm_start)
    # mixtures' probabilies
    probs = np.empty((chunks_no, 2), dtype = np.float64)
    means = probs.copy() 
    sds   = probs.copy() # mixtures' standard deviations.
    x_percentiles = np.empty(chunks_no, dtype = np.float64)

    # NOTE: the control "x" does not appear here
    # s, e      indices of the are being fitted
    # ss, se    indices used to decide upon denoising
    for i, (s, ss, se, e) in enumerate(overlapped_percentile_pairs(len(x), chunks_no)):
        g_mix.fit(y[s:e])
        signal[ss:se] = g_mix.is_signal(y[ss:se])
        probs[i,:] = g_mix.probabilities()
        means[i,:] = g_mix.means()
        x_percentiles[i] = x[ss]
        sds[i,:] = g_mix.standard_deviations()
    return signal, probs, means, sds, x_percentiles



class GaussianMixtureSpline(Spline):
    """Fit spline to data denoised by a mixture model.

    Fit spline to data denoised on subsequent, quantile-defined chunks of data.
    Each chunk is denoised individually by a two-component gaussian mixture model.

    """
    def fit(self, x, y,
            chunks_no=20,
            warm_start=True,
            drop_duplicates_and_sort=True,
            **kwds):
        """Fit a denoised spline."""
        assert chunks_no > 0
        self.chunks_no = int(chunks_no)
        self.set_xy(x, y, drop_duplicates_and_sort)
        x, y = self.x, self.y
        x.shape = x.shape[0], 1
        y.shape = y.shape[0], 1
        self.signal, self.probs, self.means, self.sds, self.x_percentiles = \
            fit_interlapping_mixtures(x, y,
                                      self.chunks_no,
                                      warm_start,
                                      sort = False)
        x_signal = self.x[self.signal].ravel()
        y_signal = self.y[self.signal].ravel()
        self.spline = beta_spline(x = x_signal,
                                  y = y_signal,
                                  chunks_no = self.chunks_no)

    def is_signal(self, x_new, y_new):
        """Denoise the new data."""
        i = np.searchsorted(self.x_percentiles, x_new) - 1
        return np.abs(self.medians[i] - y_new) <= self.stds[i] * self.std_cnt

    def predict(self, x):
        return self.spline(x).reshape(-1, 1)

    def fitted(self):
        return self.spline(self.x.ravel()).reshape(-1, 1)

    def __repr__(self):
        """Represent the model."""
        #TODO make this more elaborate.
        return "This is a spline model with gaussian mixture denoising."

