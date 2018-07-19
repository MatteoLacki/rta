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
    # NOTE: the control "x" does not appear here
    # s, e      indices of the are being fitted
    # ss, se    indices used to decide upon denoising
    for i, (s, ss, se, e) in enumerate(overlapped_percentile_pairs(len(x), chunks_no)):
        g_mix.fit(y[s:e])
        idxs = signal_idx, noise_idx = np.argsort(g_mix.covariances_.ravel()) # signal's variance is small
        signal[ss:se] = g_mix.predict(y[ss:se]) == signal_idx
        probs[i,:] = g_mix.weights_[idxs]
        means[i,:] = g_mix.means_.ravel()[idxs]
        covariances[i,:] = g_mix.covariances_.ravel()[idxs]
    return signal, probs, means, covariances


class GaussianMixtureSpline(Model):
    """A general class for this sort of silly things."""
    def df_2_data(self, data, x_name='x', y_name='y'):
        """Prepare data, if not ready."""
        data = data.drop_duplicates(subset=x_name, keep=False, inplace=False)
        data = data.sort_values([x_name, y_name])
        self.x, self.y = (np.asarray(data[name]).reshape(-1,1) 
                          for name in (x_name, y_name))

    def process_input(self, x, y, chunks_no=20):
        assert chunks_no > 0
        assert x is not None 
        assert y is not None
        self.chunks_no = int(chunks_no)
        self.x, self.y = x, y

    def fit(self, x, y,
            chunks_no=20,
            warm_start=True,
            **kwds):
        """Fit a denoised spline."""
        self.process_input(x, y, chunks_no)
        signal, self.probs, self.means, self.covariances = \
            fit_interlapping_mixtures(self.x, self.y, self.chunks_no, self.warm_start)
        # x_signal = self.x[signal].ravel()
        # y_signal = self.y[signal].ravel()
        # self.spline = beta_spline(x = x_signal,
        #                           y = y_signal,
        #                           chunks_no = self.chunks_no)
        self.spline = beta_spline(x = self.x[signal],
                                  y = self.x[signal],
                                  chunks_no = self.chunks_no)
        self.signal = signal.reshape(-1, 1)

    def predict(self, x):
        return self.spline(x).reshape(-1, 1)

    def fitted(self):
        return self.spline(self.x.ravel()).reshape(-1, 1)

    def __repr__(self):
        """Represent the model."""
        #TODO make this more elaborate.
        return "This is a GMM_OLS combo class for super-duper fitting."


    def plot(self,
             knots_no = 1000,
             plt_style = 'dark_background',
             show = True,
             fence = True,
             fence_color = 'gold'):
        """Plot the spline.
        Args:
            knots_no (int):    number of points used to plot the fitted spline?
            plt_style (str):   the matplotlib style used for plotting.
            fence_color (str): the color of the fence around signal region.
            show (logical):    show the plot immediately. Alternatively, add some more elements on the canvas before using it.
        """
        super().plot(knots_no, plt_style, show=False)
        if fence:
            plot_signal_fence(self.x_percentiles,
                              self.signal_regions[:,0],
                              self.signal_regions[:,1],
                              color = fence_color,
                              show  = show)
        else:
            # making show that we see a plot if we don't want and a fence
            # and want to see it :)
            if show:
                plt.show()


    def gaussian_mixture_spline(x, y,
                                chunks_no       = 20,
                                warm_start      = True,
                                drop_duplicates = True,
                                sort            = True,
                                folds           = None,
                                fold_stats      = (mae, mad),
                                model_stats     = (np.mean, np.median, np.std)):
        m = GaussianMixtureSpline()
        m.fit(x, y, chunks_no, warm_start, drop_duplicates, sort)
        if folds is not None:
            m.cv(folds, fold_stats, model_stats)
    return m

