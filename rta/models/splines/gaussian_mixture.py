"""Gaussian mixture based denoising. """
import numpy as np
from sklearn.mixture import GaussianMixture as GM

from rta.array_operations.misc import percentiles, percentile_pairs_of_N_integers as percentile_pairs
from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.splines.spline import Spline
from rta.models.splines.beta_splines import beta_spline



def fit_interlapping_mixtures(x, y, chunks_no=20, warm_start=True):
    """Fit mixtures on overlapping sets of data."""
    signal = np.empty(len(x), dtype=np.bool_)
    g_mix = GM(n_components = 2, warm_start=warm_start)
    probs = np.empty((chunks_no, 2), dtype=np.float64)
    means = probs.copy()
    covariances = probs.copy()
    x_percentiles = np.empty(chunks_no, dtype=np.float64)

    # NOTE: the control "x" does not appear here
    # s, e      indices of the are being fitted
    # ss, se    indices used to decide upon denoising
    for i, (s, ss, se, e) in enumerate(overlapped_percentile_pairs(len(x), chunks_no)):
        g_mix.fit(y[s:e])
        idxs = signal_idx, noise_idx = np.argsort(g_mix.covariances_.ravel()) # signal's variance is small
        signal[ss:se] = g_mix.predict(y[ss:se]) == signal_idx
        probs[i,:] = g_mix.weights_[idxs]
        means[i,:] = g_mix.means_.ravel()[idxs]
        x_percentiles[i] = x[ss]
        covariances[i,:] = g_mix.covariances_.ravel()[idxs]
    return signal, probs, means, covariances, x_percentiles


def get_inflection_point(sd_signal,
                         sd_noise,
                         prob_signal,
                         prob_noise):
    """This function will compute points of equal density of the two components."""
    pass


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
        self.signal, self.probs, self.means, self.covariances, self.x_percentiles = \
            fit_interlapping_mixtures(x, y, self.chunks_no, warm_start)
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

