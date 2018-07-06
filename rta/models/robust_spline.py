"""The Robust Spline class.

The Robust Spline performs median based denoising using windowing,
and then fits a beta spline using least squares.
"""

import numpy as np
import pandas as pd

from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.GMLSQSpline import GMLSQSpline
from rta.models.splines.beta_splines import beta_spline
from rta.stats.stats import mad, mae, confusion_matrix

def mad_window_filter(x, y, chunks_no=100, sd_cnt=3, x_sorted=False):
    """Some would say, this is madness.

    But this is 'Robust' Statistics!
    """
    if not x_sorted:
        assert all(x[i] <= x[i+1] for i in range(len(x)-1)), \
            "If 'x' ain't sorted, than I don't believe that 'y' is correct."
    signal  = np.empty(len(x),      dtype=np.bool_)
    medians = np.empty(chunks_no,   dtype=np.float64)
    stds    = np.empty(chunks_no,   dtype=np.float64)
    x_percentiles = np.empty(chunks_no, dtype=np.float64)

    scaling = 1.4826

    # NOTE: the control "x" does not appear herek
    # s, e      indices of the are being fitted
    # ss, se    indices used to decide upon denoising
    for i, (s, ss, se, e) in enumerate(overlapped_percentile_pairs(len(x), chunks_no)):
        __mad, median = mad(y[s:e], return_median=True)
        medians[i] = median
        stds[i] = sd = scaling * __mad
        x_percentiles[i] = x[ss]
        signal[ss:se] = np.abs(y[ss:se] - median) < sd * sd_cnt
    return signal, medians, stds, x_percentiles



class RobustSpline(GMLSQSpline):
    def adjust(self, x, y):
        """Remove dupilcate x entries. Sort by x."""
        d = pd.DataFrame({'x':x, 'y':y})
        d = d.drop_duplicates(subset='x', keep=False)
        d = d.sort_values(['x', 'y'])
        return d.x.values, d.y.values

    def fit(self, x, y,
            chunks_no=20,
            std_cnt=3,
            adjust=True):
        """Fit a denoised spline."""
        assert chunks_no > 0
        assert std_cnt > 0
        assert len(x) == len(y)
        self.chunks_no = int(chunks_no)
        self.std_cnt = int(std_cnt)
        self.x, self.y = self.adjust(x, y) if adjust else (x, y)
        self.signal, self.medians, self.stds, self.x_percentiles = \
            mad_window_filter(self.x,
                              self.y,
                              self.chunks_no,
                              self.std_cnt,
                              x_sorted = True)
        self.spline = beta_spline(self.x[self.signal],
                                  self.y[self.signal],
                                  self.chunks_no)

    # what about the corner conditions? 
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
        #TODO make this more elaborate.
        return "This is a SQSpline class for super-duper fitting."

    # TODO get rid of params and move it up the object ladder
    def cv(self, folds,
                 fold_stats = (mae, mad),
                 model_stats= (np.mean, np.median, np.std),
                 confusion  = True,
                 *pass_through_args):
        """Run cross-validation."""
        assert len(self.x) == len(folds)
        if confusion:
            self.fit(x, y, chunks_no, std_cnt)
            signal_fold_free = self.signal.copy()

        m_stats = []
        cv_out = []
        for fold in np.unique(folds):
            x_train = x[folds != fold]
            y_train = y[folds != fold]
            x_test  = x[folds == fold]
            y_test  = y[folds == fold]
            n = SQSpline()
            n.fit(x_train,
                  y_train,
                  self.chunks_no,
                  self.std_cnt,
                  adjust=False)
            errors = np.abs(n.predict(x_test) - y_test)
            n_signal = n.is_signal(x_test, y_test)
            stats = [stat(errors) for stat in fold_stats]
            m_stats.append(stats)
            cm = confusion_matrix(m.signal[d_run.fold == fold], n_signal)
            cv_out.append((n, stats, cm))

        m_stats = np.array(m_stats)
        m_stats = np.array([stat(m_stats, axis=0) for stat in model_stats])
        m_stats = pd.DataFrame(m_stats)
        m_stats.columns = ["fold_" + fs.__name__ for fs in fold_stats]
        m_stats.index = [ms.__name__ for ms in model_stats]

        return (m_stats, cv_out, self.chunks_no) + tuple(pass_through_args)
