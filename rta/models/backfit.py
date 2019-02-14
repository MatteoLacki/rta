try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np

from rta.misc import plot_matrix_sizes
from rta.models.model import Model


class Backfit(Model):
    def __init__(self, model, k):
        # get enough copies of the models.
        # we could as well set up an infinite generator
        # and cut it.
        self.models = [model.copy() for _ in range(k)]

    def fit(self, x, y):
        self.x = x
        self.y = y
        for m in self.models:
            m.fit(x, y)
            y = m.res()

    def __call__(self, x):
        out = np.zeros(x.shape)
        for m in self.models:
            out += m(x)
        return out

    def plot_all(self, plt_style='dark_background',
                       show=True,
                       shared_selection=True,
                       **kwds):
        """Plot all consecutive fitting results of the backfitting.
    
        Args:
            plt_style (str): The style of the matplotlib visualization [default 'dark_background'].
                             Check https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
            show (bool): Show the figure, or just add it to the canvas [default True].
            shared_selection (boolean): Should the selection in one window work for all [default True].
            kwds: optional keyword arguments for the 'plot' functions of the underlying models.
        """
        if plt:
            plt.style.use(plt_style)
            rows_no, cols_no = plot_matrix_sizes(len(self.models))
            i = 1
            for m in self.models:
                if i == 1:
                    ax1 = plt.subplot(rows_no, cols_no, i)
                else:
                    if shared_selection:
                        plt.subplot(rows_no, cols_no, i, sharex=ax1, sharey=ax1)
                    else:
                        plt.subplot(rows_no, cols_no, i)
                m.plot(plt_style=plt_style, show=False, **kwds)
                i += 1
            if show:
                plt.show()
        else:
            print('Install matplotlib to use this function.')
            raise ModuleNotFoundError