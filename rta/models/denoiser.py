try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np
from scipy.signal import order_filter

from rta.array_operations.dedupy import dedup_np
from rta.models.model import Model
from rta.math.splines import beta as beta_spline


class Denoiser(Model):
    def __call__(self, x, y=None):
        """Return classifications of points into signal or noise.

        Args:
            x (np.array): The control variable.
        """
        L = self.L(x)
        U = self.U(x)
        if y is None:
            return L, U
        else:
            return np.logical_and(L <= y, y <= U)

    def plot(self, plt_style='dark_background',
                   show=True,
                   nodes=1000,
                   points=True,
                   noise_c = 'violet',
                   signal_c = 'yellow',
                   **kwds):
        if plt:
            plt.style.use(plt_style)
            if points:
                colors = np.full(self.x.shape, noise_c)
                colors[self(self.x, self.y)] = signal_c
                plt.scatter(self.x, self.y, c=colors, **kwds)
            if nodes:
                xs = np.linspace(min(self.x), max(self.x), nodes)
                ls, us = self(xs)
                plt.plot(xs, ls, c='orange')
                plt.plot(xs, us, c='orange')
            if show:
                plt.show()
        else:
            print('Install matplotlib to use this function.')
            raise ModuleNotFoundError



class DenoiserRollingOrder(Denoiser):
    def __init__(self, w=np.ones(41), l=4, u=37, n=100):
        """Constructor.
        
        Args:
            w (odd int): The moving wi(n)dow.
            n (int): the number of nodes used for the beta spline (roughly correspond to 100/k-percentiles).
            i (int): which ordered statistics to calculate.
        """
        assert l < u, "l should be smaller than u."
        self.w = w
        self.n = n
        self.u = u
        self.l = l
        self.params = {'w':w, 'n':n, 'l':l, 'u':u}

    def fit(self, x, y, sort=True):
        if sort:
            i = np.argsort(x)
            x, y = x[i], y[i]
        self.x = x
        self.y = y
        x, y = dedup_np(x, y)
        self.ll = order_filter(y, self.w, self.l)
        self.uu = order_filter(y, self.w, self.u)
        self.L = beta_spline(x, self.ll, self.n)
        self.U = beta_spline(x, self.uu, self.n)