"""The Robust Spline class.

The Robust Spline performs median based denoising using windowing,
and then fits a beta spline using least squares.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.splines.spline import Spline
from rta.models.splines.beta_splines import beta_spline
from rta.stats.stats import mad

def mad_window_filter(x, y, chunks_no=100, sd_cnt=3, x_sorted=False):
    """Filter based on median absolute deviation.

    Estimates both the mead and standard deviation of the signal normal
    distribution using the robust estimates: median and mad.
    Repeat this in a sliding window strategy.
    The data is divided into 'chunks_no'-quantiles bins in 'x' variable.
    The fitting is performed based on 3 bins at a time.
    Results are projected on the middle bin.
    The calculation proceeds in a sliding window fashion.
    Corner cases (first and last quantile) fit to two consecutive bins.
    
    Args:
        x (np.array) 1D control
        y (np.array) 1D response
        chunks_no (int) The number of quantile bins.
        sd_cnt (float) How many standard deviations are considered to be within signal range.
        x_sorted (logical) Are 'x' and 'y' sorted with respect to 'x'.
    Returns:
        signal (np.array of logical values) Is the point considered to be a signal?
        medians (np.array) Estimates of medians in consecutive bins.
        stds (np.array) Estimates of standard deviations in consecutive bins.
        x_percentiles (np.array) Knots of the spline fitting, needed to filter out noise is 'is_signal'.        
    """
    if not x_sorted:
        assert all(x[i] <= x[i+1] for i in range(len(x)-1)), \
            "If 'x' ain't sorted, than I don't believe that 'y' is correct."
    signal  = np.empty(len(x),      dtype=np.bool_)
    medians = np.empty(chunks_no,   dtype=np.float64)
    stds    = np.empty(chunks_no,   dtype=np.float64)
    x_percentiles = np.empty(chunks_no, dtype=np.float64)

    scaling = 1.4826

    # NOTE: the control "x" does not appear here
    # s, e      indices of the are being fitted
    # ss, se    indices used to decide upon denoising
    for i, (s, ss, se, e) in enumerate(overlapped_percentile_pairs(len(x), chunks_no)):
        __mad, median = mad(y[s:e], return_median=True)
        medians[i] = median
        stds[i] = sd = scaling * __mad
        x_percentiles[i] = x[ss]
        signal[ss:se] = np.abs(y[ss:se] - median) <= sd * sd_cnt
    return signal, medians, stds, x_percentiles


class RobustSpline(Spline):
    def fit(self, x, y,
            chunks_no=20,
            std_cnt=3,
            drop_duplicates_and_sort=True):
        """Fit a robust spline.
        Args:
            x (np.array) 1D control
            y (np.array) 1D response
            chunks_no (int) The number of quantile bins.
            drop_duplicates_and_sort (logical) Should we drop duplicates in 'x' and sort 'x' and 'y' w.r.t. 'x'?
        """
        assert chunks_no > 0
        assert std_cnt > 0
        self.chunks_no = int(chunks_no)
        self.std_cnt = int(std_cnt)
        self.set_xy(x, y, drop_duplicates_and_sort)

        self.signal, self.medians, self.stds, self.x_percentiles = \
            mad_window_filter(self.x,
                              self.y,
                              self.chunks_no,
                              self.std_cnt,
                              x_sorted = True)
        self.spline = beta_spline(self.x[self.signal],
                                  self.y[self.signal],
                                  self.chunks_no)

    def is_signal(self, x_new, y_new):
        """Denoise the new data."""
        i = np.searchsorted(self.x_percentiles, x_new) - 1
        return np.abs(self.medians[i] - y_new) <= self.stds[i] * self.std_cnt

    def predict(self, x):
        return self.spline(x)

    def fitted(self):
        return self.spline(self.x.ravel())

    def __repr__(self):
        """Represent the model."""
        fit = hasattr(self, 'signal')
        cv = hasattr(self, 'fold_stats')
        return "This is a RobustSpline super-duper fitting.\n\tFitted\t\t\t{}\n\tCross-validated\t\t{}".format(fit, cv)
