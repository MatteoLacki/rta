import numpy as np
from scipy.interpolate import LSQUnivariateSpline as Spline
from sklearn.mixture import GaussianMixture as GM

from rta.array_operations.misc import percentiles, percentile_pairs_of_N_integers as percentile_pairs
from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.GMLSQSpline import GMLSQSpline


class RapidGMLSQSpline(GMLSQSpline):
    def fit(self,
            x=None,
            y=None,
            chunks_no=20,
            warm_start=True,
            **kwds):
        """Fit a denoised spline."""
        x, y, chunks_no = self.check_input(x, y, chunks_no)
        # first fit a gaussian mixture
        g_mix = GM(n_components = 2, warm_start=warm_start)
        g_mix.fit(y)
        idxs = signal_idx, noise_idx = np.argsort(g_mix.covariances_.ravel()) # signal's variance is small
        signal = g_mix.predict(y) == signal_idx
        