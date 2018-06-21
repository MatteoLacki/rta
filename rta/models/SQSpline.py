import numpy as np
from numpy import logical_and as AND

from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.GMLSQSpline import GMLSQSpline, fit_spline


def mean_absolute_deviation(x):
    return np.abs(x).mean()

def mad(x, return_median=False):
    """Compute median absolute deviation (from median, of course) and median."""
    median = np.median(x)
    if return_median:
        return np.median(np.abs(x - median)), median
    else:
        return np.median(np.abs(x - median))


def mad_window_filter(x, y, chunks_no=100, sd_cnt=3, x_sorted=False):
    """Some would say, this is madness.

    But this is 'Robust' Statistics!
    """
    if not x_sorted:
        assert all(x[i] <= x[i+1] for i in range(len(x)-1)), \
            "If 'x' ain't sorted, than I don't believe that 'y' is correct."
    signal = np.empty(len(x), dtype=np.bool_)
    medians = np.empty(chunks_no, dtype=np.float64)
    st_devs = np.empty(chunks_no, dtype=np.float64)
    x_percentiles = np.empty(chunks_no, dtype=np.float64)
    scaling = 1.4826
    # NOTE: the control "x" does not appear here
    # s, e      indices of the are being fitted
    # ss, se    indices used to decide upon denoising
    for i, (s, ss, se, e) in enumerate(overlapped_percentile_pairs(len(x), chunks_no)):
        __mad, median = mad(y[s:e], return_median=True)
        medians[i] = median
        st_devs[i] = sd = scaling * __mad
        x_percentiles[i] = x[ss]
        signal[ss:se] = np.abs(y[ss:se] - median) < sd * sd_cnt
    return signal, medians, st_devs, x_percentiles


class SQSpline(GMLSQSpline):
    def df_2_data(self, data, x_name='x', y_name='y'):
        """Prepare data, if not ready.

        We don't use any Gaussian Mixture model here.
        So, we don't need to reshape the data to the silly format.
        """
        self.data = data.drop_duplicates(subset=x_name, keep=False, inplace=False)
        self.data = self.data.sort_values([x_name, y_name])
        self.has_data = True

    def fit(self, x, y, chunks_no=20, sd_cnt=3, **kwds):
        """Fit a denoised spline."""
        assert chunks_no > 0
        assert sd_cnt > 0
        self.chunks_no = int(chunks_no)
        self.sd_cnt = int(sd_cnt)
        self.signal, self.medians, self.st_devs, self.x_percentiles = mad_window_filter(x, y, chunks_no, sd_cnt, True)
        self.spline = fit_spline(x[self.signal], y[self.signal], chunks_no)

    # what about the corner conditions? 
    def is_signal(self, x_new, y_new):
        """Denoise the new data."""
        i = np.searchsorted(self.x_percentiles, x_new) - 1
        return np.abs(self.medians[i] - y_new) <= self.st_devs[i] * self.sd_cnt

    def predict(self, x):
        return self.spline(x)

    def fitted(self):
        return self.spline(self.x.ravel())

    def __repr__(self):
        """Represent the model."""
        #TODO make this more elaborate.
        return "This is a SQSpline class for super-duper fitting."