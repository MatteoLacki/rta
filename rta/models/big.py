try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np

from rta.array_operations.iterators import xyg_iter
from rta.plot.multi import multiplot


class BigModel(object):
    def __init__(self, models):
        """Constuctor.
        Args:
            m (dict of callables): mapping between runs and the coordinate models to be fitted.
        """
        self.models = models

    def fit(self, x, y, g):
        self.x = np.array(x)
        self.y = np.array(y)
        self.g = np.array(g)
        for X, Y, gr in xyg_iter(x, y, g):
            self.models[gr].fit(X, Y - X)

    def __call__(self, x, g):
        x = np.array(x)
        g = np.array(g)
        x_new = np.zeros(x.shape)
        for gr, model in self.models.items():
            X_gr = x[g == gr]
            x_new[g == gr] = X_gr + model(X_gr)
        return x_new

    def res(self):
        """Return the residuals."""
        return self.fitted() - self.x

    def fitted(self):
        """Return the aligned retention times."""
        return self(self.x, self.g)

    def plot(self, show=True, 
                   shared_selection=True,
                   shape=None,
                   residuals=False,
                   **kwds):
        """Plot all fitting results."""
        if plt:
            if residuals:
                plots = (lambda: mo.plot_residuals(show=False, **kwds)
                         for gr, mo in self.models.items())
            else:
                plots = (lambda: mo.plot(show=False, **kwds)
                         for gr, mo in self.models.items())
            multiplot(plots,
                      len(self.models), 
                      None,
                      shared_selection,
                      show)
        else:
            print('Install matplotlib to use this function.')
            raise ModuleNotFoundError


