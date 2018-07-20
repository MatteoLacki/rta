"""Gaussian mixture based denoising. """
import matplotlib.pyplot as plt
import numpy as np

from rta.array_operations.misc import percentiles, percentile_pairs_of_N_integers as percentile_pairs
from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.denoising.window_based import sort_by_x
from rta.models.mixtures.two_component_gaussian_mixture import TwoComponentGaussianMixture as GM
from rta.models.splines.spline import Spline
from rta.models.splines.beta_splines import beta_spline
from rta.plotters.plot_signal_fence import plot_signal_fence
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
        x (np.array): 1D control
        y (np.array): 1D response
        chunks_no (int): The number of quantile bins.
        warm_start (logical): should consecutive gaussian mixture models start from the previously estimated values of paramters. Speeds up computations, but sometimes produces nonsensical values.
        sort (logical): Are 'x' and 'y' sorted with respect to 'x'.
    Returns:
        signal (np.array of logical values): Is the point considered to be a signal?
        medians (np.array): Estimates of medians in consecutive bins.
        stds (np.array): Estimates of standard deviations in consecutive bins.
        x_percentiles (np.array): Knots of the spline fitting, needed to filter out noise is 'is_signal'.
        signal_regions (np.array):

    """
    x, y = sort_by_x(x, y) if sort else (x, y)
    signal = np.empty(len(x), dtype = np.bool_)
    gm = GM(warm_start = warm_start)
    # mixtures' probabilies
    probs = np.empty((chunks_no, 2), dtype = np.float64)
    means = probs.copy() 
    sds   = probs.copy() # mixtures' standard deviations.
    signal_regions = probs.copy() # points where densities times mixture probabilities equalize.
    x_percentiles = np.empty(chunks_no + 1, dtype = np.float64)

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
    x_percentiles[i+1] = x[se] # the maximal value
    return signal, probs, means, sds, x_percentiles, signal_regions



class GaussianMixtureSpline(Spline):
    """Fit spline to data denoised by a mixture model.

    Fit spline to data denoised on subsequent, quantile-defined chunks of data.
    Each chunk is denoised individually by a two-component gaussian mixture model.

    """

    def __init__(self,
                 chunks_no  = 20,
                 warm_start = True):
        """Initialize the class.

        Similarly to SKLearn, pass the fitting arguments here.

        Args:
            chunks_no (int) The number of quantile bins.
            warm_start (logical): should consecutive gaussian mixture models start from the previously estimated values of paramters. Speeds up computations, but sometimes produces nonsensical values.
        """
        assert chunks_no > 0
        self.chunks_no   = int(chunks_no)
        self.warm_start  = warm_start

    def copy(self):
        """Copy constructor."""
        return GaussianMixtureSpline(self.chunks_no, self.warm_start)

    def fit(self, x, y,
            drop_duplicates = True,
            sort            = True):
        """Fit a denoised spline.

        Args:
            x (np.array): 1D control
            y (np.array): 1D response
            sort (logical): Are 'x' and 'y' sorted with respect to 'x'.
        """
        self.set_xy(x, y, drop_duplicates, sort)
        self.signal, self.probs, self.means, self.sds,\
        self.x_percentiles, self.signal_regions = \
            fit_interlapping_mixtures(self.x,
                                      self.y,
                                      self.chunks_no,
                                      self.warm_start,
                                      sort = False)
        self.spline = beta_spline(self.x[self.signal],
                                  self.y[self.signal],
                                  self.chunks_no)

    def is_signal(self, x, y):
        """Denoise the new data.

        Args:
            x (np.array of floats): x-coordinates of points to classify as signal or noise.
            y (np.array of floats): y-coordinates of points to classify as signal or noise.
        
        Returns:
            np.array of logicals: should the points be considered signal?
        """
        i = np.searchsorted(self.x_percentiles, x) - 1
        # check, if signal is within the borders of the algorithm.
        in_range = (i > -1) & (i < self.chunks_no)
        is_signal = np.full(shape      = x.shape,
                            fill_value = False,
                            dtype      = np.bool_)
        i = i[in_range]
        y = y[in_range]
        bottom = self.signal_regions[i,0] # bottom-line values for y
        top    = self.signal_regions[i,1] # top-line values for y
        is_signal[in_range] = (bottom <= y) & (y <= top)
        return is_signal

    def __repr__(self):
        """Represent the model."""
        #TODO make this more elaborate.
        return "This is a spline model with gaussian mixture denoising."

    def plot(self,
             knots_no = 1000,
             plt_style = 'dark_background',
             show = True,
             fence = True,
             medians = True,
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
                              show = False)
        if medians:
            x = self.x_percentiles
            plt.hlines(y = self.means[:,0],
                       xmin = x[0:-1],
                       xmax = x[1:],
                       color= 'gold',
                       linestyles = 'dashed')
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
    """Fit the gaussian mixture spline.

    Args:
        x (np.array): 1D control
        y (np.array): 1D response
        chunks_no (int): The number of quantile bins.
        warm_start (logical): should consecutive gaussian mixture models start from the previously estimated values of paramters. Speeds up computations, but sometimes produces nonsensical values.
        drop_duplicates (logical): Drop duplicates in 'x' in both 'x' and 'y' arrays. 
        sort (logical): Sort 'x' and 'y' w.r.t. 'x'.
        folds (np.array of ints): Assignments of data points to folds to test model's generalization capabilities.
        folds_stats (tuple of functions): Statistics of the absolute values of errors on the test sets.
        model_stats (tuple of functions): Statistics of fold statistics.

    Returns:
        GaussianMixtureSpline: a fitted instance of 'GaussianMixtureSpline'.
    """
    m = GaussianMixtureSpline(chunks_no, warm_start)
    m.fit(x, y, drop_duplicates, sort)
    if folds is not None:
        m.cv(folds, fold_stats, model_stats)
    return m