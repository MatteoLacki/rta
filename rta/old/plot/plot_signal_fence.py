import matplotlib.pyplot as plt


def plot_signal_fence(x, bottom, top,
                      color = 'gold',
                      show  = True):
    """Plot a fence arond a region defined by horizontal and vertical lines.

    Args:
        x (np.array of floats):      x-coordinates of the top and bottom lines.
        bottom (np.array of floats): y-coordinates of the bottom lines.
        top (np.array of floats):    y-coordinates of the top lines.
        color (string):              The color of the fence.
        show (logical): show the plot immediately. Alternatively, add some more elements on the canvas before using it.
    """
    assert len(x) == (len(bottom) + 1) == (len(top) + 1)
    plt.hlines(y=bottom,    xmin=x[0:-1],        xmax=x[1:],       colors=color)
    plt.hlines(y=top,       xmin=x[0:-1],        xmax=x[1:],       colors=color)
    plt.vlines(x=x[1:-1],   ymin=top[1:],        ymax=top[:-1],    colors=color)
    plt.vlines(x=x[1:-1],   ymin=bottom[1:],     ymax=bottom[:-1], colors=color)
    plt.vlines(x=x[[0,-1]], ymin=bottom[[0,-1]], ymax=top[[0,-1]], colors=color)
    if show:
        plt.show()
