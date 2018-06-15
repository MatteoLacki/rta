import numpy as np
from scipy.interpolate import LSQUnivariateSpline as Spline
from sklearn.mixture import GaussianMixture as GM

from rta.array_operations.misc import percentiles, percentile_pairs_of_N_integers as percentile_pairs
from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.GMLSQSpline import GMLSQSpline, fit_spline


class RapidGMLSQSpline(GMLSQSpline):
    def fit(self,
            x=None,
            y=None,
            chunks_no=20,
            warm_start=True,
            **kwds):
        """Fit a denoised spline."""
        x, y, chunks_no = self.check_input(x, y, chunks_no)
        
        # fit a gaussian mixture to all of the response
        g_mix = GM(n_components=2,
                   warm_start=warm_start)
        g_mix.fit(y)
        idxs = signal_idx, noise_idx = np.argsort(g_mix.covariances_.ravel()) # signal's variance is small
        signal = g_mix.predict(y) == signal_idx

        # fit a spline
        x_signal = x[signal].ravel()
        y_signal = y[signal].ravel()
        spline = fit_spline(x_signal, y_signal, chunks_no)
        res = (y_signal - spline(x_signal)).reshape(-1, 1)

        # fit a gaussian mixture to the residuals
        g_mix.fit(res)
        idxs = signal_idx, noise_idx = np.argsort(g_mix.covariances_.ravel()) # signal's variance is small
        signal[signal] = g_mix.predict(res) == signal_idx

        # fit the final spline
        x_signal = x[signal].ravel()
        y_signal = y[signal].ravel()
        self.spline = fit_spline(x_signal, y_signal, chunks_no)
        self.signal = signal.reshape(-1, 1)