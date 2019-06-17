import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

from rta.models.interpolant import Interpolant
from rta.models.spline import Spline
from rta.array_operations.dedupy import dedup_np
from rta.math.splines import beta as beta_spline


class RollingMedian(Interpolant):
    """The rolling median interpolator.

    Idea is as straight as a hair of a Mongol: get rid of the noise by
    fitting a roling median and interpolate every other k-th median.
    Of course, since we calculate all other medians too, we could get more
    playful with their evaluation.
    """
    def __init__(self, ws=51, k=10):
        """Constructor.
        
        Args:
            ws (odd int): window size.
            k (int): each k-th median will be used for interpolation
        """
        self.ws = ws
        self.k = k
        self.params = {'ws':ws, 'k':k}

    def __repr__(self):
        return "RollingMedian(ws:{} k:{})".format(self.ws, self.k)

    def fit(self, x, y, sort=True, **kwds):
        """Fit the model.

        Args:
            x (np.array): The control variable.
            y (np.array): The response variable.
        """
        if sort:
            i = np.argsort(x)
            x, y = x[i], y[i]
        self.medians = medfilt(y, self.ws)
        self.interpo = interp1d(x[::self.k],
                                self.medians[::self.k],
                                bounds_error=False,
                                fill_value=0,
                                **kwds)
        self.x = x
        self.y = y


#TODO: implement this.
class RolllingMedianSimple(RollingMedian):
    """Avoid calculating too many medians."""
    def fit(self, x, y, sort=True):
        """Fit the model.

        Args:
            x (np.array): The control variable.
            y (np.array): The response variable.
        """
        pass



class RollingMedianSpline(Spline):
    """The rolling median spline."""
    def __init__(self, ws=51, n=100):
        """Constructor.
        
        Args:
            ws (odd int): window size.
            n (int): the number of nodes used for the beta spline (roughly correspond to 100/k-percentiles).
        """
        self.ws = ws
        self.n = n
        self.params = {'ws':ws, 'n':n} # this is for copy to work

    def __repr__(self):
        return "RollingMedianSpline(ws:{})".format(self.ws)

    def fit(self, x, y, sort=True, dedup=True, **kwds):
        if sort:
            i = np.argsort(x)
            x, y = x[i], y[i]
        self.x = x
        self.y = y
        x, y = dedup_np(x, y)
        self.medians = medfilt(y, self.ws)
        self.spline = beta_spline(x, self.medians, self.n, **kwds)
