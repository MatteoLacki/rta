"""Gaussian mixture based denoising. """

import numpy as np

from rta.array_operations.misc import percentiles, percentile_pairs_of_N_integers as percentile_pairs
from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.denoising.window_based import sort_by_x
from rta.models.mixtures.two_component_gaussian_mixture import TwoComponentGaussianMixture as GM
from rta.models.splines.spline import Spline
from rta.models.splines.beta_splines import beta_spline
from rta.stats.stats import mad, mae

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
    gm = GM(warm_start = warm_start)
    # mixtures' probabilies
    probs = np.empty((chunks_no, 2), dtype = np.float64)
    means = probs.copy() 
    sds   = probs.copy() # mixtures' standard deviations.
    signal_regions = probs.copy() # points where densities times mixture probabilities equalize.
    x_percentiles = np.empty(chunks_no, dtype = np.float64)

    # NOTE: the control "x" does not appear here
    # s, e      indices of the are being fitted
    # ss, se    indices used to decide upon denoising
    for i, (s, ss, se, e) in enumerate(overlapped_percentile_pairs(len(x), chunks_no)):
        gm.fit(y[s:e])
        x_percentiles[i] = x[ss]
        # all the info we can get from a two-component gaussian mixture model
        signal[ss:se]       = gm.is_signal(y[ss:se])
        probs[i,:]          = gm.probabilities()
        means[i,:]          = gm.means()
        sds[i,:]            = gm.standard_deviations()
        signal_regions[i,:] = gm.signal_region()
    return signal, probs, means, sds, x_percentiles, signal_regions



class GaussianMixtureSpline(Spline):
    """Fit spline to data denoised by a mixture model.

    Fit spline to data denoised on subsequent, quantile-defined chunks of data.
    Each chunk is denoised individually by a two-component gaussian mixture model.

    """
    def fit(self, x, y,
            chunks_no       = 20,
            warm_start      = True,
            drop_duplicates = True,
            sort            = True,
            **kwds):
        """Fit a denoised spline."""
        assert chunks_no > 0
        self.chunks_no = int(chunks_no)
        self.set_xy(x, y, drop_duplicates, sort)
        x, y = self.x, self.y
        x.shape = x.shape[0], 1
        y.shape = y.shape[0], 1
        self.signal, self.probs, self.means, self.sds,\
        self.x_percentiles, self.signal_regions = \
            fit_interlapping_mixtures(x, y,
                                      self.chunks_no,
                                      warm_start,
                                      sort = False)
        x_signal = self.x[self.signal].ravel()
        y_signal = self.y[self.signal].ravel()
        self.spline = beta_spline(x = x_signal,
                                  y = y_signal,
                                  chunks_no = self.chunks_no)

    def is_signal(self, x, y):
        """Denoise the new data."""
        i = np.searchsorted(self.x_percentiles, x)-1
        # check, if signal is within the borders of the algorithm.
        in_range = (i > -1) & (i < self.chunks_no - 1)
        is_signal = np.full(shape      = x.shape,
                            fill_value = False,
                            dtype      = np.bool_)
        i = i[in_range]
        y = y[in_range]
        bottom = self.signal_regions[i,0] # bottom-line values for y
        top    = self.signal_regions[i,1] # top-line values for y
        is_signal[in_range] = (bottom <= y) & (y <= top)
        return is_signal

    def predict(self, x):
        return self.spline(x).reshape(-1, 1)

    def fitted(self):
        return self.spline(self.x.ravel()).reshape(-1, 1)

    def __repr__(self):
        """Represent the model."""
        #TODO make this more elaborate.
        return "This is a spline model with gaussian mixture denoising."

    def plot(self,
             knots_no = 1000,
             plt_style = 'dark_background',
             show = True, 
             **kwds):
        """Plot the spline.

        Args:
            knots_no (int):  number of points used to plot the fitted spline?
            plt_style (str): the matplotlib style used for plotting.
            show (logical):  show the plot immediately. Alternatively, add some more elements on the canvas before using it.
        """
        super().plot(knots_no, plt_style, show=False)
        plt.scatter()
        if show:
            plt.show()



def gaussian_mixture_spline(x, y,
                            chunks_no=20,
                            warm_start=True,
                            drop_duplicates_and_sort=True,
                            folds=None,
                            fold_stats  = (mae, mad),
                            model_stats = (np.mean, np.median, np.std)):
    m = GaussianMixtureSpline()
    m.fit(x, y, chunks_no, warm_start, drop_duplicates_and_sort)
    if folds is not None:
        m.cv(folds, fold_stats, model_stats)
    return m