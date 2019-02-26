try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from rta.misc import plot_matrix_sizes



# for some reason cannot pass the stupid list
def multiplot(plot_callables,
              plots_no,
              shape=None,
              shared_selection=True,
              show=True):
    """Place multiple plots on the canvas.

    Args:
        plot_callables (iterator of callables): each call f() makes a matplotlib plot
        plots_no: the number of plots.
        shape (tuple): number of rows and columns in the multiplot
        shared_selection (boolean): Should the selection in one window work for all [default True].
        show (bool): Show the figure, or just add it to the canvas [default True]
    """
    rows_no, cols_no = plot_matrix_sizes(plots_no) if shape is None else shape
    i = 1
    for p in plot_callables:
        if i == 1:
            ax1 = plt.subplot(rows_no, cols_no, i)
        else:
            if shared_selection:
                plt.subplot(rows_no, cols_no, i, sharex=ax1, sharey=ax1)
            else:
                plt.subplot(rows_no, cols_no, i)
        p()
        i += 1
    if show:
        plt.show()
