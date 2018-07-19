"""The Robust Spline class.

The Robust Spline performs median based denoising using windowing,
and then fits a beta spline using least squares.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.denoising.window_based import sort_by_x
from rta.models.splines.spline import Spline
from rta.models.splines.beta_splines import beta_spline
from rta.plotters.plot_signal_fence import plot_signal_fence
from rta.stats.stats import mad, mae


def mad_window_filter(x, y,
                      chunks_no = 100,
                      sd_cnt    = 3,
                      sort      = True):
    """Filter based on median absolute deviation.

    Estimates both the mead and standard deviation of the signal normal
    distribution using the robust estimates: median and mad.
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
        sd_cnt (float) How many standard deviations are considered to be within signal range.
        x_sorted (logical) Are 'x' and 'y' sorted with respect to 'x'.
    Returns:
        signal (np.array of logical values) Is the point considered to be a signal?
        medians (np.array) Estimates of medians in consecutive bins.
        sts (np.array) Estimates of standard deviations in consecutive bins.
        x_percentiles (np.array) Knots of the spline fitting, needed to filter out noise is 'is_signal'.        
    """
    x, y = sort_by_x(x, y) if sort else (x, y)
    signal  = np.empty(len(x),    dtype = np.bool_)
    medians = np.empty(chunks_no, dtype = np.float64)
    sds     = np.empty(chunks_no, dtype = np.float64)
    x_percentiles = np.empty(chunks_no + 1, dtype=np.float64)

    scaling = 1.4826

    # NOTE: the control "x" does not appear here
    # s, e      indices of the are being fitted
    # ss, se    indices used to decide upon denoising
    for i, (s, ss, se, e) in enumerate(overlapped_percentile_pairs(len(x), chunks_no)):
        __mad, median = mad(y[s:e], return_median = True)
        medians[i] = median
        sds[i] = sd = scaling * __mad
        x_percentiles[i] = x[ss]
        signal[ss:se] = np.abs(y[ss:se] - median) <= sd * sd_cnt
    x_percentiles[i+1] = x[se] # the maximal value
    return signal, medians, sds, x_percentiles



class RobustSpline(Spline):
    def __init__(self,
                 chunks_no = 20,
                 sd_cnt    = 3):
        """Constructor."""
        assert chunks_no > 0
        assert sd_cnt > 0
        self.chunks_no = int(chunks_no)
        self.sd_cnt = int(sd_cnt)

    def copy(self):
        """Copy constructor."""
        return RobustSpline(self.chunks_no, self.sd_cnt)

    def fit(self, x, y,
            drop_duplicates = True,
            sort            = True):
        """Fit a robust spline.
        
        Args:
            x (np.array): 1D control
            y (np.array): 1D response
            chunks_no (int): The number of quantile bins.
            sd_cnt (float): The number of standard deviations beyond which points are considered noise.
            drop_duplicates_and_sort (logical) Drop duplicates in 'x' and sort 'x' and 'y' w.r.t. 'x'?
        """
        self.set_xy(x, y, drop_duplicates, sort)
        self.signal, self.medians, self.sds, self.x_percentiles = \
            mad_window_filter(self.x,
                              self.y,
                              self.chunks_no,
                              self.sd_cnt,
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
        is_signal[in_range] = np.abs(self.medians[i] - y) <=\
                              self.sds[i] * self.sd_cnt
        return is_signal

    def __repr__(self):
        """Represent the model."""
        fit = hasattr(self, 'signal')
        cv = hasattr(self, 'fold_stats')
        return "This is a RobustSpline super-duper fitting.\n\tFitted\t\t\t{}\n\tCross-validated\t\t{}".format(fit, cv)

    def plot(self,
             knots_no    = 1000,
             plt_style   = 'dark_background',
             show        = True,
             fence       = True,
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
            bottom = self.medians - self.sds * self.sd_cnt
            top =    self.medians + self.sds * self.sd_cnt
            plot_signal_fence(self.x_percentiles,
                              bottom,
                              top,
                              color = fence_color,
                              show  = show)
        else:
            # making show that we see a plot if we don't want and a fence
            # and want to see it :)
            if show:
                plt.show()



def robust_spline(x, y,
                  chunks_no       = 20,
                  sd_cnt          = 3,
                  drop_duplicates = True,
                  sort            = True,
                  folds           = None,
                  fold_stats      = (mae, mad),
                  model_stats     = (np.mean, np.median, np.std)):
    """Fit the robust spline.

    Args:
        x (np.array): 1D control
        y (np.array): 1D response
        chunks_no (int): The number of quantile bins.
        sd_cnt (float): The number of standard deviations beyond which points are considered noise.
        drop_duplicates (logical): Drop duplicates in 'x' in both 'x' and 'y' arrays. 
        sort (logical): Sort 'x' and 'y' w.r.t. 'x'.
        folds (np.array of ints): Assignments of data points to folds to test model's generalization capabilities.
        folds_stats (tuple of functions): Statistics of the absolute values of errors on the test sets.
        model_stats (tuple of functions): Statistics of fold statistics.

    Returns:
        RobustSpline: a fitted instance of 'RobustSpline'.
    """
    m = RobustSpline(chunks_no, sd_cnt)
    m.fit(x, y, drop_duplicates, sort)
    if folds is not None:
        m.cv(folds, fold_stats, model_stats)
    return m