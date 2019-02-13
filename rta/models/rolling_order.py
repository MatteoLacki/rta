try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import order_filter

from rta.array_operations.dedupy import dedup_np
from rta.math.splines import beta as beta_spline
from rta.models.interpolant import Interpolant
from rta.models.spline import Spline


class RollingOrder(Interpolant):
    """The rolling median interpolator."""
    def __init__(self, w=np.ones(41), k=10, i=21):
        """Constructor.
        
        Args:
            w (odd int): The moving wi(n)dow.
            k (int): each k-th median will be used for interpolation
            i (int): which ordered statistics to calculate.
        """
        self.w = w
        self.k = k
        self.i = i
        self.params = {'w':w, 'k':k, 'i':i}

    def __repr__(self):
        return "RollingOrder({})".format(self.i)

    def fit(self, x, y, sort=True):
        """Fit the model.

        Args:
            x (np.array): The control variable.
            y (np.array): The response variable.
        """
        if sort:
            i = np.argsort(x)
            x, y = x[i], y[i]
        self.order = order_filter(y, self.w, self.i)
        self.interpo = interp1d(x[::self.k],
                                self.order[::self.k],
                                bounds_error=False,
                                fill_value=0)
        self.x = x
        self.y = y



class RollingOrderSpline(Spline):
    """The rolling median interpolator."""
    def __init__(self, w=np.ones(41), n=100, i=21):
        """Constructor.
        
        Args:
            w (odd int): The moving wi(n)dow.
            n (int): the number of nodes used for the beta spline (roughly correspond to 100/k-percentiles).
            i (int): which ordered statistics to calculate.
        """
        self.w = w
        self.n = n
        self.i = i
        self.params = {'w':w, 'n':n, 'i':i}

    def __repr__(self):
        return "RollingOrderSpline({}, {})".format(self.n, self.i)

    def fit(self, x, y, sort=True):
        """Fit the model.

        Args:
            x (np.array): The control variable.
            y (np.array): The response variable.
        """
        if sort:
            i = np.argsort(x)
            x, y = x[i], y[i]
        self.x = x
        self.y = y
        x, y = dedup_np(x, y)
        self.order = order_filter(y, self.w, self.i)
        self.spline = beta_spline(x, self.order, self.n)
