import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

from rta.models.model import Model


class RollingMedian(Model):
    """The rolling median interpolator.

    Idea is as straight as a hair of a Mongol: get rid of the noise by
    fitting a roling median and interpolate every other k-th median.
    Of course, since we calculate all other medians too, we could get more
    playful with their evaluation.
    TODO: add a wrapper around this method.
    """
    def __init__(self, ws=51, k=10):
        """Constructor.
        
        Args:
            ws (odd int): window size.
            k (int): each k-th median will be used for interpolation
        """
        self.ws = ws
        self.k = k

    def __repr__(self):
        return "RollingMedian(ws:{} k:{})".format(self.ws, self.k)

    def fit(self, x, y, sort=False):
        if sort:
            i = np.argsort(x)
            x, y = x[i], y[i]
        self.medians = medfilt(y, self.ws)
        self.interpo = interp1d(x[::self.k],
                                self.medians[::self.k],
                                bounds_error=False,
                                fill_value=0)
        self.x = x
        self.y = y

    def __call__(self, x):
        return self.interpo(x)

    def plot(self, nodes=1000, plt_style='dark_background', show=True):
        plt.style.use(plt_style)
        plt.scatter(self.x, self.y, s=1)
        xs = np.linspace(min(self.x), max(self.x), nodes)
        plt.plot(xs, self(xs), c='orange')
        if show:
            plt.show()

# rmi = RollingMedian()
# rmi.fit(x, y-x)
# rmi.plot()
# plt.axhline(y=0, color='red')
# plt.scatter(x, y - x - rmi(x), s=1)
# plt.show()