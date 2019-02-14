try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from rta.misc import plot_matrix_sizes


def plot_distances_to_reference(X,
                                plt_style='dark_background',
                                shared_selection=True,
                                show=True,
                                **kwds):
    """Plot distances to the reference run.

    Args:
        X (pd.DataFrame): DataFrame with columns x, y (reference), and run.    
        plt_style (str): The style of the matplotlib visualization [default 'dark_background'].
                         Check https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
        show (bool): Show the figure, or just add it to the canvas [default True].
        shared_selection (boolean): Should the selection in one window work for all [default True].
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
