import numpy as np
from scipy.interpolate import LSQUnivariateSpline as Spline
from sklearn.mixture import GaussianMixture as GM

from rta.array_operations.misc import percentiles, percentile_pairs_of_N_integers as percentile_pairs
from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.base_model import Model


def fit_interlapping_mixtures(x, y, chunks_no=20, warm_start=True):
    """Fit mixtures on overlapping sets of data."""
    signal = np.empty(len(x), dtype=np.bool_)
    g_mix = GM(n_components = 2, warm_start=warm_start)
    probs = np.empty((chunks_no, 2), dtype=np.float64)
    means = probs.copy()
    covariances = probs.copy()
    # NOTE: the control "x" does not appear here
    # s, e      indices of the are being fitted
    # ss, se    indices used to decide upon denoising
    for i, (s, ss, se, e) in enumerate(overlapped_percentile_pairs(len(x), chunks_no)):
        g_mix.fit(y[s:e])
        idxs = signal_idx, noise_idx = np.argsort(g_mix.covariances_.ravel()) # signal's variance is small
        signal[ss:ee] = g_mix.predict(y[ss:ee]) == signal_idx
        probs[i,:] = g_mix.weights_[idxs]
        means[i,:] = g_mix.means_.ravel()[idxs]
        covariances[i,:] = g_mix.covariances_.ravel()[idxs]
    return signal, probs, means, covariances



def fit_spline(x, y, chunks_no=20):
    """Efficienlty fit spline to the denoined data with the least squares method."""
    x_inner_percentiles = percentiles(x, chunks_no, inner=True)
    return Spline(x, y, x_inner_percentiles)



class GMLSQSpline(Model):
    """A general class for this sort of silly things."""
    def __init__(self):
        self.has_data = False

    def df_2_data(self, data, x_name='x', y_name='y'):
        """Prepare data, if not ready."""
        data = data.drop_duplicates(subset=x_name, keep=False, inplace=False)
        data = data.sort_values([x_name, y_name])
        self.x, self.y = (np.asarray(data[name]).reshape(-1,1) 
                          for name in (x_name, y_name))
        self.has_data = True

    def check_input(self, x, y, chunks_no=20):
        assert chunks_no > 0
        chunks_no = int(chunks_no)
        if not self.has_data:
            assert not x is None or not y is None
            self.x, self.y = x, y
        else:
            x, y = self.x, self.y
        self.chunks_no = chunks_no
        return x, y, chunks_no

    def fit(self,
            x=None,
            y=None,
            chunks_no=20,
            warm_start=True,
            **kwds):
        """Fit a denoised spline."""
        x, y, chunks_no = self.check_input(x, y, chunks_no)
        signal, self.probs, self.means, self.covariances = fit_interlapping_mixtures(x, y, chunks_no, warm_start)
        x_signal = self.x[signal].ravel()
        y_signal = self.y[signal].ravel()
        self.spline = fit_spline(x_signal, y_signal, chunks_no)
        self.signal = signal.reshape(-1, 1)

    def predict(self, x):
        return self.spline(x).reshape(-1, 1)

    def fitted(self):
        return self.spline(self.x.ravel()).reshape(-1, 1)

    def res(self):
        """Get residuals."""
        return self.y - self.fitted()

    def coef(self):
        return self.spline.get_coeffs()

    def __repr__(self):
        """Represent the model."""
        #TODO make this more elaborate.
        return "This is a GMM_OLS combo class for super-duper fitting."

