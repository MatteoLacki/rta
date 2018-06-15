import numpy as np

from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.GMLSQSpline import GMLSQSpline, fit_spline


def mad_median(x):
    """Compute median absolute deviation (from median, of course) and median."""
    median = np.median(x)
    return np.median(np.abs(x - median)), median


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
    scaling = 1.4826
    # NOTE: the control "x" does not appear here
    # s, e      indices of the are being fitted
    # ss, se    indices used to decide upon denoising
    for i, (s, ss, se, e) in enumerate(overlapped_percentile_pairs(len(x), chunks_no)):
        mad, median = mad_median(y[s:e])
        medians[i] = median
        st_devs[i] = sd = scaling * mad
        signal[ss:se] = np.abs(y[ss:se] - median) < sd * sd_cnt
    return signal, medians, st_devs


class SQSpline(GMLSQSpline):
    def df_2_data(self, data, x_name='x', y_name='y'):
        """Prepare data, if not ready.

        We don't use any Gaussian Mixture model here.
        So, we don't need to reshape the data to the silly format.
        """
        data = data.drop_duplicates(subset=x_name, keep=False, inplace=False)
        data = data.sort_values([x_name, y_name])
        self.x, self.y = (np.asarray(data[name]) for name in (x_name, y_name))
        self.has_data = True

    def fit(self, x=None, y=None, chunks_no=20, sd_cnt=3, **kwds):
        """Fit a denoised spline."""
        x, y, chunks_no = self.check_input(x, y, chunks_no)
        self.signal, self.medians, self.st_devs = mad_window_filter(x, y, chunks_no, sd_cnt, True)
        self.spline = fit_spline(x[self.signal], y[self.signal], chunks_no)