import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

from rta.models.base_model import Model


class RollingMedianInterpolation(Model):
    def __init__(self, ws=51, k=10):
        """Constructor.
        
        Args:
            ws (odd int): window size.
            k (int): each k-th median will be used for interpolation
        """
        self.ws = ws
        self.k = k

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

# rmi = RollingMedianInterpolation()
# rmi.fit(x, y-x)
# rmi.plot()

# should we recalculate the median or not?
# will recalculating it ease the effect of zeros?
    # of course not!
    # it all boils down to points being closer.

# plt.axhline(y=0, color='red')
# plt.scatter(x, y - x - rmi(x), s=1)
# plt.show()