"""Align towards one chosen reference retention time.

To align differently, have a look at Tenzerization.
"""

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np

from rta.misc import plot_matrix_sizes


class Aligner(object):
    def __init__(self, models):
        """Constuctor.
        Args:
            m (dict of callables): mapping between runs and the coordinate models to be fitted.
        """
        self.m = models

    def fit(self, X):
        """Fit coordinate models to the distances to the reference.

        Args:
            X (pd.DataFrame): dataframe with columns run, x (values to align), y (reference values).
        """
        self.X = X
        for r, Xr in X.groupby('run'):
            # we model the distances to the reference values!
            self.m[r].fit(Xr.x.values, Xr.y.values - Xr.x.values)

    def __repr__(self):
        return "Aligner"

    def __call__(self, X):
        """Align the observations in X.

        Args:
            X (pd.DataFrame): dataframe with columns runs and x,
            where x are the values to be aligned.
        Returns:
            np.array: aligned retention times, for each pair (run, x) specified in X.
        """
        runs = X.run.unique()
        x_new = np.zeros((len(X),))
        for r in runs:
            x = X.x[X.run == r]
            x_new[X.run == r] = x + self.m[r](x)
        return x_new

    def res(self):
        """Return the residuals."""
        return self.fitted() - self.X.x.values

    def fitted(self):
        """Return the aligned retention times."""
        return self(self.X)

    def plot(self, plt_style='dark_background',
                   show=True, 
                   shared_selection=True,
                   residuals=False,
                   **kwds):
        """Plot fitting results.
    
        Args:
            plt_style (str): The style of the matplotlib visualization [default 'dark_background'].
                             Check https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
            show (bool): Show the figure, or just add it to the canvas [default True].
            shared_selection (boolean): Should the selection in one window work for all [default True].
            residuals (boolean): Should the plot contain the residuals instead of the fitting?
            kwds: optional keyword arguments for the 'plot' functions of the underlying models.
        """
        if plt:
            plt.style.use(plt_style)
            rows_no, cols_no = plot_matrix_sizes(len(self.m))
            i = 1
            for run, model in self.m.items():
                if i == 1:
                    ax1 = plt.subplot(rows_no, cols_no, i)
                else:
                    if shared_selection:
                        plt.subplot(rows_no, cols_no, i, sharex=ax1, sharey=ax1)
                    else:
                        plt.subplot(rows_no, cols_no, i)
                if residuals:
                    model.plot_residuals(plt_style=plt_style, show=False, **kwds)
                else:
                    model.plot(plt_style=plt_style, show=False, **kwds)
                i += 1
            if show:
                plt.show()
        else:
            print('Install matplotlib to use this function.')
            raise ModuleNotFoundError

    def plot_residuals(self, plt_style='dark_background',
                             show=True, 
                             shared_selection=True,
                             **kwds):
        """Plot fitting results.
    
        This function is introduced to retain similarity to the Model class.

        Args:
            plt_style (str): The style of the matplotlib visualization [default 'dark_background'].
                             Check https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
            show (bool): Show the figure, or just add it to the canvas [default True].
            shared_selection (boolean): Should the selection in one window work for all [default True].
            kwds: optional keyword arguments for matplotlib.plt
        """
        self.plot(plt_style, show, shared_selection, True, **kwds)