try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np

from rta.array_operations.iterators import xyg_iter
from rta.misc import plot_matrix_sizes
from rta.plot.multi import multiplot


def simple_xy_plot(x, y,
                   plt_style='dark_background',
                   show=True,
                   label="",
                   **kwds):
    """Plot distances to the reference run.

    Args:
        x (np.array): x values
        y (np.array): y values
        plt_style (str): The style of the matplotlib visualization [default 'dark_background'].
                         Check https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
        show (bool): Show the figure, or just add it to the canvas [default True].
        kwds: optional keyword arguments for the 'plot' functions of the underlying models.
    """
    plt.scatter(x, y-x, **kwds)
    plt.plot((0,max(x)), (0,0), c='orange')
    plt.title(label)
    if show:
        plt.show()



def plot_distances_to_reference(x, y,
                                g=None,
                                shared_selection=True,
                                shape=None,
                                show=True,
                                **kwds):
    """Plot distances to the reference run for a pandas dataframe.

    Args:
        x (np.array): x values
        y (np.array): y values
        g (np.array): grouping variable
        shared_selection (boolean): Should the selection in one window work for all [default True].
        show (bool): Show the figure, or just add it to the canvas [default True].
        kwds: optional keyword arguments for the 'plot' functions of the underlying models.
    """
    if g is None:
        g = np.full(x.shape, "")
    plot_callables = (lambda: simple_xy_plot(X, Y, label=gr, show=False, **kwds)
                      for X, Y, gr in xyg_iter(x,y,g))
    plots_no = len(np.unique(g))
    multiplot(plot_callables, plots_no, shape, show, shared_selection)



def plot_distances_to_reference_pd(X,
                                  plt_style='dark_background',
                                  shared_selection=True,
                                  show=True,
                                  grouped=False,
                                  **kwds):
    """Plot distances to the reference run for a pandas dataframe.

    Args:
        X (pd.DataFrame): DataFrame with columns x, y (reference), and run.    
        plt_style (str): The style of the matplotlib visualization [default 'dark_background'].
                         Check https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
        shared_selection (boolean): Should the selection in one window work for all [default True].
        show (bool): Show the figure, or just add it to the canvas [default True].
        kwds: optional keyword arguments for the 'plot' functions of the underlying models.
    """
    plt.style.use(plt_style)
    rows_no, cols_no = plot_matrix_sizes(len(X.run.unique()))
    i = 1
    for r, d in X.groupby('run'):
        x = d.x.values
        y = d.y.values
        if i == 1:
            ax1 = plt.subplot(rows_no, cols_no, i)
        else:
            if shared_selection:
                plt.subplot(rows_no, cols_no, i, sharex=ax1, sharey=ax1)
            else:
                plt.subplot(rows_no, cols_no, i)
        plt.scatter(x, y-x, label=r, **kwds)
        plt.plot((0,max(x)), (0,0), c='orange')
        i += 1
    if show:
        plt.show()
